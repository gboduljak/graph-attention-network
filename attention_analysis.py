from typing import List, Tuple
import numpy as np
from scipy.stats import entropy
import torch
import networkx as nx
from models import GATWithLogging


def extract_neighbourhood_attention_entropies(model: GATWithLogging, graph: nx.Graph, edge_index: torch.Tensor,
                                              head: int, layer: int, num_nodes: int) -> Tuple[List[float], List[float]]:
  neighborhood_dist_entropy_list = []
  uniform_dist_entropy_list = []
  edge_src_nodes = edge_index[0].cpu().numpy()
  edge_tgt_nodes = edge_index[1].cpu().numpy()
  layer_attention_scores = model.get_attention_scores(layer)
  layer_attention_scores = layer_attention_scores.squeeze(dim=-1).cpu().numpy()

  for node in range(num_nodes):
    node_neighbourhood_mask = edge_tgt_nodes == node
    neighbourhood_attention_distribution = layer_attention_scores[head, node_neighbourhood_mask].flatten()
    # neighborhood attention must sum to 1
    assert (np.isclose(neighbourhood_attention_distribution.sum(), 1))
    # selected neighbourhood must match actual neighbourhood in the graph
    neighbourhood_nodes = edge_src_nodes[node_neighbourhood_mask]
    assert (set(neighbourhood_nodes) == set(graph.neighbors(node)))
    # compute uniform distribution corresponding to neighbourhood size
    neighborhood_size = len(neighbourhood_attention_distribution)
    corresponding_ideal_uniform_distribution = np.ones(neighborhood_size) / neighborhood_size
    # log entropies of actual attention distribution and theoretical uniform distribution
    neighborhood_dist_entropy_list.append(entropy(neighbourhood_attention_distribution, base=2))
    uniform_dist_entropy_list.append(entropy(corresponding_ideal_uniform_distribution, base=2))

  return (neighborhood_dist_entropy_list, uniform_dist_entropy_list)
