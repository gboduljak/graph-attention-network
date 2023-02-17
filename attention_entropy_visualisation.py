from typing import List
import torch
import matplotlib.pyplot as plt
from attention_analysis import extract_neighbourhood_attention_entropies
from attention_entropy_visualisation_utils import draw_entropy_head_plot, draw_entropy_heads_plot
from models import GATWithLogging
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data
from enum import Enum
from device import device


class AttentionVisualisationMode(Enum):
  PER_HEAD = 0,
  PER_LAYER = 1


def visualise_attention_entropy(model: GATWithLogging, batched_graph: Data, heads_per_layer: List[int],
                                vis_mode: AttentionVisualisationMode):
  graph_nx = to_networkx(batched_graph)
  model.eval()

  with torch.no_grad():
    (X, edge_index) = (batched_graph.x.to(device), batched_graph.edge_index.to(device))
    num_nodes, _ = X.shape
    model.forward(X, edge_index)

    for layer in range(len(model.layers)):
      layer_attention_entropies = []
      layer_uniform_entropies = []

      layer_attention_scores = model.get_attention_scores(layer)
      layer_attention_scores = layer_attention_scores.squeeze(dim=-1).cpu().numpy()

      num_heads, _ = layer_attention_scores.shape
      num_heads = heads_per_layer[layer] if heads_per_layer else num_heads

      for head in range(num_heads):
        head_neighborhood_entropies, head_uniform_dist_entropies = extract_neighbourhood_attention_entropies(
            model, graph_nx, edge_index, head, layer, num_nodes)

        if vis_mode == AttentionVisualisationMode.PER_HEAD:
          fig = draw_entropy_head_plot(plt.gca(), head_neighborhood_entropies, head_uniform_dist_entropies,
                                       f'attention head={head}, layer={layer}')
          plt.show()

        layer_attention_entropies.append(head_neighborhood_entropies)
        layer_uniform_entropies.append(head_uniform_dist_entropies)

      if vis_mode == AttentionVisualisationMode.PER_LAYER:
        if num_heads <= 4:
          fig = draw_entropy_heads_plot(layer_attention_entropies, layer_uniform_entropies, layer, (1, num_heads))
        else:
          rows = 2
          cols = num_heads // 2
          fig = draw_entropy_heads_plot(layer_attention_entropies, layer_uniform_entropies, layer, (rows, cols))
          fig.subplots_adjust(top=0.9)
          fig.set_size_inches(19.5, 11.5)
        plt.show()