from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from layers import GATLayer


class GATWithLogging(nn.Module):

  def __init__(self):
    super().__init__()
    self.layers = []

  def get_attention_scores(self, layer: int) -> torch.Tensor:
    pass


class GATv1PPI(GATWithLogging):

  def __init__(self, dim_in: int, num_classes: int):
    super().__init__()
    self.model_name = 'GATv1PPI'
    # assumes self loops were added by dataset transform
    self.layers = nn.ModuleList([
        GATLayer(dim_in, 256, 4, act=F.elu, reduce='concat'),
        GATLayer(1024, 256, 4, act=F.elu, reduce='concat', skip=True),
        GATLayer(1024, num_classes, 6, act=nn.Identity(), reduce='avg', skip=True)
    ])
    self.reset_parameters()

  def get_attention_scores(self, layer: int) -> torch.Tensor:
    return self.layers[layer].get_attention_scores()

  def get_embeddings(self, layer: int = 0) -> torch.Tensor:
    return self.layers[layer].get_embeddings()

  def forward(self, X: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
    out = X
    for layer in self.layers:
      out = layer(out, edge_index)
    return out

  def reset_parameters(self):
    for layer in self.layers:
      layer.reset_parameters()


class GCNPPI(nn.Module):

  def __init__(self, dim_in: int, num_classes: int):
    super().__init__()
    self.model_name = 'GCNPPI'
    # assumes self loops were added by dataset transform
    self.fst_layer = [
        gnn.GCNConv(in_channels=dim_in, out_channels=256, activation=nn.Identity(), bias=False, add_self_loops=False)
        for _ in range(4)
    ]
    self.snd_layer = [
        gnn.GCNConv(in_channels=1024, out_channels=256, activation=nn.Identity(), bias=False, add_self_loops=False)
        for _ in range(4)
    ]
    self.last_layer = [
        gnn.GCNConv(in_channels=1024,
                    out_channels=num_classes,
                    activation=nn.Identity(),
                    bias=False,
                    add_self_loops=False) for _ in range(6)
    ]
    self.skip_to_snd = [nn.Linear(1024, 256, False) for _ in range(4)]
    self.skip_to_last = [nn.Linear(1024, num_classes, False) for _ in range(6)]
    self.layers = self.fst_layer + self.snd_layer + self.last_layer
    self.skips = self.skip_to_snd + self.skip_to_last

  def concat(self, out_preacts_per_head: List[torch.tensor], act):
    out_per_head = [act(out) for out in out_preacts_per_head]
    out_cat = torch.cat(out_per_head, dim=-1)
    return out_cat

  def avg(self, out_preacts_per_head: List[torch.tensor], act):
    out_preacts_stack = torch.stack(out_preacts_per_head)
    out_preacts_avg = torch.mean(out_preacts_stack, dim=0)
    out_avg = act(out_preacts_avg)
    return out_avg

  def get_embeddings(self, layer: int = 0) -> torch.Tensor:
    embeddings = [self.fst_embeddings, self.snd_embeddings, self.last_embeddings]
    return embeddings[layer]

  def forward(self, X: torch.tensor, edge_index: torch.tensor) -> torch.tensor:

    out_fst_per_head = [head(X, edge_index) for head in self.fst_layer]
    out_fst_cat = self.concat(out_fst_per_head, F.elu)
    self.fst_embeddings = out_fst_cat

    out_snd_per_head = [head(out_fst_cat, edge_index) for head in self.snd_layer]
    out_snd_skip_per_head = [linear(out_fst_cat) for linear in self.skip_to_snd]
    out_snd_per_head_res = [
        head_out + head_skip for (head_out, head_skip) in zip(out_snd_per_head, out_snd_skip_per_head)
    ]
    out_snd_cat = self.concat(out_snd_per_head_res, F.elu)
    self.snd_embeddings = out_snd_cat

    out_last_per_head = [head(out_snd_cat, edge_index) for head in self.last_layer]
    out_last_skip_per_head = [linear(out_snd_cat) for linear in self.skip_to_last]
    out_last_per_head_res = [
        head_out + head_skip for (head_out, head_skip) in zip(out_last_per_head, out_last_skip_per_head)
    ]
    out_last = self.avg(out_last_per_head_res, nn.Identity())
    self.last_embeddings = out_last

    return out_last

  def reset_parameters(self):
    for layer in self.layers:
      layer.reset_parameters()
    for layer in self.skips:
      layer.reset_parameters()