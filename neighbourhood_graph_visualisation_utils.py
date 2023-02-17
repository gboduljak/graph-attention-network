import matplotlib.pyplot as plt
import networkx as nx
from neighbourhood_analysis import NeighbourhoodAttentionStats


def draw_node_neighbourhood(node: int,
                            neighbourhood_stats: NeighbourhoodAttentionStats,
                            axis: plt.Axes,
                            plot_title: str = None):
  edge_thickness_factor = 3
  plot_node_size = 600
  plot_cmap = plt.cm.PuBu

  graph = neighbourhood_stats.neighbourhood_subgraph
  neighbourhood_size = len(graph.nodes())
  visualisation_layout = nx.spring_layout(graph)
  nx.draw_networkx_nodes(graph,
                         visualisation_layout,
                         node_color=range(neighbourhood_size),
                         cmap=plot_cmap,
                         node_size=plot_node_size,
                         ax=axis)
  nx.draw_networkx_edges(graph,
                         visualisation_layout,
                         graph.edges,
                         width=neighbourhood_stats.attention_scores * edge_thickness_factor,
                         arrowstyle='-',
                         edge_cmap=plot_cmap,
                         ax=axis)
  axis.set_title(plot_title if plot_title else f'neighbourhood attention at node={node}')
