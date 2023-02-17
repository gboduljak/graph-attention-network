import matplotlib.pyplot as plt

from enum import Enum
from typing import List
from torch_geometric.data import Data
from models import GATWithLogging
from neighbourhood_analysis import extract_neighbourhood_stats
from neighbourhood_graph_visualisation_utils import draw_node_neighbourhood


class NeighbourhoodVisualisationMode(Enum):
  PER_NODE = 0,
  PER_HEAD = 1


def visualise_neighbourhood_for(model: GATWithLogging, batched_graph: Data, layer: int, head: int, target_node: int,
                                axis: plt.Axes):
  neighbourhood_stats = extract_neighbourhood_stats(model, batched_graph, layer, head, target_node)
  max_attention = neighbourhood_stats.max_attention_score
  min_attention = neighbourhood_stats.min_attention_score
  print(f'neighbourhood stats for node={target_node}: ')
  print(f'\tmax attention: {max_attention:.4f}, min attention: {min_attention:.4f}')
  draw_node_neighbourhood(target_node, neighbourhood_stats, axis)


def visualise_neighbourhoods_for(model: GATWithLogging, batched_graph: Data, layer: int, head: int,
                                 target_nodes: List[int], vis_mode: NeighbourhoodVisualisationMode):

  num_nodes = len(target_nodes)

  if vis_mode == NeighbourhoodVisualisationMode.PER_HEAD:
    fig, axs = plt.subplots(1, num_nodes)
    fig.suptitle(f'neighbourhood for attention head={head}, layer={layer}')
    fig.subplots_adjust(top=0.9)
    fig.set_size_inches(19.5, 5.75)

    for (axis_ix, node) in enumerate(target_nodes):
      visualise_neighbourhood_for(model, batched_graph, layer, head, node, axs[axis_ix])
  else:
    for node in target_nodes:
      fig, axs = plt.subplots(1, 1)
      fig.subplots_adjust(top=0.9)
      fig.set_size_inches(4.875, 5.75)
      visualise_neighbourhood_for(model, batched_graph, layer, head, node, axs)