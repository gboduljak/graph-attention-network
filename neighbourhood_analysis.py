import torch
import numpy as np
import networkx as nx

from dataclasses import dataclass
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from models import GATWithLogging
from device import device


@dataclass
class NeighbourhoodAttentionStats:
  max_attention_score: float
  min_attention_score: float
  attention_scores: np.ndarray
  neighbourhood_subgraph: nx.Graph


def extract_neighbourhood_stats(model: GATWithLogging, batched_graph: Data, layer: int, head: int,
                                node: int) -> NeighbourhoodAttentionStats:
  graph_nx = to_networkx(batched_graph)
  model.eval()
  with torch.no_grad():
    (node_feats, edge_index) = (batched_graph.x.to(device), batched_graph.edge_index.to(device))

    model.forward(node_feats, edge_index)
    edge_tgt_nodes = edge_index[1].cpu().numpy()
    edge_src_nodes = edge_index[0].cpu().numpy()

    # extract layer attention
    layer_attention_scores = model.get_attention_scores(layer)
    layer_attention_scores = layer_attention_scores.squeeze(dim=-1).cpu().numpy()

    # extract attention distribution for node neighbourhood
    node_neighbourhood_mask = edge_tgt_nodes == node
    neighbourhood_nodes = edge_src_nodes[node_neighbourhood_mask]
    neighbourhood_attention_distribution = layer_attention_scores[head, node_neighbourhood_mask].flatten()
    max_attention = neighbourhood_attention_distribution.max()
    min_attention = neighbourhood_attention_distribution.min()

    # neighborhood attention must sum to 1
    assert (np.isclose(neighbourhood_attention_distribution.sum(), 1))
    # selected neighbourhood must match actual neighbourhood in the graph
    assert (set(neighbourhood_nodes) == set(graph_nx.neighbors(node)))

    neighbourhood_subgraph = nx.DiGraph()
    neighbourhood_subgraph.add_weighted_edges_from([
        (neighbour, node, attn) for (neighbour, attn) in zip(neighbourhood_nodes, neighbourhood_attention_distribution)
    ])

    return NeighbourhoodAttentionStats(max_attention, min_attention, neighbourhood_attention_distribution,
                                       neighbourhood_subgraph)