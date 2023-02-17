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
    self.layers = [
        gnn.GCNConv(in_channels=dim_in, out_channels=1024, activation=nn.ReLU(), bias=False, add_self_loops=False),
        gnn.GCNConv(in_channels=1024, out_channels=1024, activation=nn.ReLU(), bias=False, add_self_loops=False),
        gnn.GCNConv(in_channels=1024,
                    out_channels=num_classes,
                    activation=nn.Identity(),
                    bias=False,
                    add_self_loops=False)
    ]
    [fst_conv, snd_conv, classifier_conv] = self.layers
    self.encoder = gnn.Sequential('x, edge_index', [(fst_conv, 'x, edge_index -> x'), (snd_conv, 'x, edge_index -> x')])
    self.classifier = classifier_conv

  def get_embeddings(self, layer: int = 0) -> torch.Tensor:
    return self.layers[layer].get_embeddings()

  def forward(self, X: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
    self.embeddings = self.encoder(X, edge_index)
    return self.classifier.forward(self.embeddings, edge_index)

  def reset_parameters(self):
    self.encoder.reset_parameters()
    self.classifier.reset_parameters()