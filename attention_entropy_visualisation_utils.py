from typing import List
import numpy as np
import matplotlib.pyplot as plt


def draw_entropy_head_plot(axis: plt.Axes, neighbourhood_entropy: np.ndarray, uniform_entropy: np.ndarray, title: str):
  cmap = plt.cm.PuBuGn
  color_attention = cmap(0.7)
  color_uniform = cmap(0.4)

  def draw_entropy_histogram(entropy_array: np.ndarray, color: str, from_uniform_distribution=False, num_bins=30):
    max_entropy = np.max(entropy_array)
    bin_width = (max_entropy / num_bins) * (1.0 if from_uniform_distribution else 0.75)
    hist_values, hist_bins = np.histogram(entropy_array, bins=num_bins)
    axis.bar(hist_bins[:num_bins], hist_values[:num_bins], width=bin_width, color=color)

  draw_entropy_histogram(uniform_entropy, color=color_uniform, from_uniform_distribution=True)
  draw_entropy_histogram(neighbourhood_entropy, color=color_attention)

  axis.set_xlabel(f'entropy bin')
  axis.set_ylabel(f'# of node neighborhoods')
  axis.legend(['uniform distribution', 'attention distribution'])
  axis.set_title(title)


def draw_entropy_heads_plot(neighbourhood_entropy_per_head: List[np.ndarray],
                            uniform_entropy_per_head: List[np.ndarray], layer: int, subplots: List[int]):

  rows, cols = subplots
  current_head = 0
  fig, axs = plt.subplots(rows, cols)

  for row in range(rows):
    for col in range(cols):
      neighbourhood_entropy = neighbourhood_entropy_per_head[current_head]
      corresponding_unif_entropy = uniform_entropy_per_head[current_head]
      if rows == 1:
        draw_entropy_head_plot(axs[col], neighbourhood_entropy, corresponding_unif_entropy,
                               f'attention head={current_head}, layer={layer}')
      else:
        draw_entropy_head_plot(axs[row, col], neighbourhood_entropy, corresponding_unif_entropy,
                               f'attention head={current_head}, layer={layer}')
      current_head += 1

  fig.suptitle(f'attention distribution entropy in layer={layer}')
  fig.subplots_adjust(top=0.9)
  fig.set_size_inches(19.5, 5.75)
  return fig