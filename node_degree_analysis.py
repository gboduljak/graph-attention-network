import networkx as nx

from itertools import groupby
from operator import itemgetter


def print_nodes_with_degs(graph: nx.Graph, skip: int = 5):
  nodes_sorted_by_deg = list(sorted(graph.out_degree, key=lambda x: x[1], reverse=True))
  nodes_grouped_by_deg = [list(node_deg_group) for _, node_deg_group in groupby(nodes_sorted_by_deg, itemgetter(1))]

  for grouping in nodes_grouped_by_deg:
    node_with_deg, *_ = grouping
    _, deg = node_with_deg
    if (deg % skip == 0):
      print(node_with_deg)