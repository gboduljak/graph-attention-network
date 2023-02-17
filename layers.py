import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, scatter
from torch_geometric.nn.inits import glorot


class GATLayer(nn.Module):

  def __init__(self,
               D_in: int,
               D_out: int,
               num_heads: int = 1,
               act=F.elu,
               dropout: float = 0.0,
               reduce='none',
               skip=False):
    super().__init__()
    self.D_in = D_in
    self.D_out = D_out
    self.N_h = num_heads
    self.act = act

    self.W = nn.Parameter(torch.zeros((num_heads, D_out, D_in)))
    self.W_skip = nn.Parameter(torch.zeros((num_heads, D_out, D_in)))

    self.A_src = nn.Parameter(torch.zeros((num_heads, D_out, 1)))
    self.A_tgt = nn.Parameter(torch.zeros((num_heads, D_out, 1)))
    self.reduce = reduce
    self.dropout = dropout
    self.skip = skip

    self.reset_parameters()

  def reset_parameters(self):
    glorot(self.W)
    glorot(self.W_skip)
    glorot(self.A_src)
    glorot(self.A_tgt)

  def get_attention_scores(self):
    return self.attention_scores

  def get_embeddings(self):
    return self.embeddings

  def forward(self, H_in: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
    edge_src = edge_index[:1].t().squeeze()
    edge_tgt = edge_index[1:].t().squeeze()

    N, _ = H_in.shape
    W = self.W
    W_skip = self.W_skip

    A_src = self.A_src
    A_tgt = self.A_tgt
    D_in = self.D_in
    D_out = self.D_out
    N_h = self.N_h
    act = self.act
    dropout = self.dropout
    training = self.training
    skip = self.skip

    W = W.view((N_h, D_out, D_in))
    W_skip = W.view((N_h, D_out, D_in))

    A_src = A_src.view((N_h, D_out))
    A_tgt = A_tgt.view((N_h, D_out))

    H_in = F.dropout(H_in, dropout, training)

    H_w = torch.einsum('ij, nkj -> nik', H_in, W)  # (N_h, |V|, D_out)
    H_w = F.dropout(H_w, dropout, training)  # (N_h, |V|, D_out)

    H_w_src = torch.index_select(H_w, 1, edge_src)  # (N_h, |E|, D_out)
    H_w_tgt = torch.index_select(H_w, 1, edge_tgt)  # (N_h, |E|, D_out)

    E_pre_src = torch.einsum('nij, nj -> ni', H_w_src, A_src)  # (N_h, |E|, 1), a_src^T Whi
    E_pre_tgt = torch.einsum('nij, nj -> ni', H_w_tgt, A_tgt)  # (N_h, |E|, 1), a_tgt^T Whj

    E_pre = E_pre_src + E_pre_tgt  # (N_h, |E|, 1), a^T [Whi || Whj]
    E = F.leaky_relu(E_pre, negative_slope=0.2)  # (N_h, |E|, 1), LeakyRelu(a^T [Whi || Whj])

    alpha_scores = softmax(E, edge_tgt, dim=1).view((N_h, *edge_src.shape, 1))  # (N_h, |E|, 1)
    alpha_scores = F.dropout(alpha_scores, dropout, training)
    Alpha = alpha_scores.repeat(1, 1, D_out)  # (N_h, |E|, D_out)

    self.attention_scores = alpha_scores

    H_out_pre = scatter(Alpha * H_w_src, edge_tgt, dim=1, reduce='sum')  # (N_h, |V|, D_out)
    if skip:
      H_skip_to_add = torch.zeros_like(H_out_pre)
      if D_in != D_out:
        # H_skip : (|V|, D_in)
        # W_skip : (N_h, D_out, D_in)
        # H_skip_add : (N_h, |V|, D_out)
        H_skip_to_add = torch.einsum('ij, nkj -> nik', H_in, W_skip)
      else:
        # H_skip_add : (N_h, |V|, D_out)
        H_skip_to_add = H_in.repeat(N_h, 1, 1)

      H_out_pre += H_skip_to_add

    if self.reduce == 'none':
      H_out = act(H_out_pre)  # (N_h, |V|, D_out)
      assert (H_out.shape == (N_h, N, D_out))
      return H_out
    elif self.reduce == 'concat':
      H_out = act(H_out_pre)
      H_out_per_head = torch.tensor_split(H_out, N_h)
      H_out_cat = torch.cat(H_out_per_head, dim=-1).squeeze()
      self.embeddings = H_out_cat
      assert (H_out_cat.shape == (N, N_h * D_out))
      return H_out_cat
    else:
      H_out_pre_avg = torch.mean(H_out_pre, dim=0)
      H_out = act(H_out_pre_avg)
      self.embeddings = H_out
      assert (H_out.shape == (N, D_out))
      return H_out
