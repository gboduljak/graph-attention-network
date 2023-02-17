# Graph Attention Networks Implementation and Attention Analysis

#### Project Description

In Section 3.4 of Graph Attention Networks [Veličković et al.](https://arxiv.org/abs/1710.10903), authors report that GATs achieve state-of-the-art performance across Cora and PPI. Since GATs were able to improve upon GCNs by a margin of 1.5% on Cora, the authors suggest that “assigning different weights to nodes of a same neighbourhood may be beneficial”. Furthermore, the authors report a significant improvement on the inductive PPI benchmark and “highlight the potential of attention-based models when dealing with arbitrarily structured graphs”. The authors also state that GATs “are able to assign different importances to different nodes within a neighbourhood while dealing with different sized neighbourhoods”, without concrete supporting evidence they actually do that on PPI. Since PPI is a protein interaction network and since those networks are usually heterophilic, I hypothesise that in such a setting, the attention is more beneficial and is unlikely to be uniform. This hypothesis could be verified by determining the probability distribution of scores learned by attention heads. The purpose of this study is to verify the hypothesis and provide evidence for the quoted claims.

This project was submitted as a part of examination for [Graph Representation Learning](https://www.cs.ox.ac.uk/teaching/courses/2022-2023/grl/) at Oxford.


#### Why custom **GATLayer** implementation?
The inductive PPI model presented in [the paper](https://arxiv.org/pdf/1710.10903.pdf) requires residual (skip) connections. In the [offical implementation](https://github.com/PetarV-/GAT/blob/master/utils/layers.py), residual connection is added to preactivations. In the course, we used PyTorch Geometric and I attempted to use it. However, [the implementation of GATConv in PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv) does not have this type of residual connection implemented. Unfortunately, it was not possible to implement this type of residual connection by simple addition. Due to constraints of this coursework, it was easier to give a "from scratch" implementation of **GATLayer** matching the one from [the paper](https://arxiv.org/pdf/1710.10903.pdf) than to implement required type of residual connection in PyTorch Geometric. In future, I might propose the implementation of this residual connection in PyTorch Geometric. Interestingly, [DGL implementation of GATConv](https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.GATConv.html#dgl.nn.pytorch.conv.GATConv) supports the required type of residual connection.

**GATLayer** implementation is in *layers.py* and the corresponding tests for that implementation are in the notebook *test_gat_layer_implementation.ipynb*.

#### How is **GATLayer** implementation tested?
In the notebook *test_gat_layer_implementation.ipynb*, outputs of **GATLayer** are compared to outputs of [PyTorch Geometric GATConv](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv), on a small example. Activation functions and reduction modes relevant for the experiment implementation are tested.

#### Model definition
In the report, analysis was conducted on the inductive PPI model presented in [the paper](https://arxiv.org/pdf/1710.10903.pdf). This model was compared to "the GCN model of
comparable size". Both of those are defined in *models.py*. 

#### Training and evaluation
Implementation of the training loop and evaluation is in *training_loop.py* and *evaluation.py*,
respectively. Models were trained on the Google Colab GPUs and the corresponding training notebooks are located in the *experiments* folder.

#### Analysis implementation and visualisations
Logic for the presented analyses is in *attention_analysis.py* and in *neighbourhood_analysis.py*. Visualisation of those are implementated in  *attention_entropy_visualisation.py* and *neighbourhood_analysis_visualisation.py*.

#### Reproducibility
In the *weights* folder, there are weights for the model analysed in the report.
By executing the notebook *reproduce_plots_seed_42.ipynb*, it is possible to reproduce the plots presented in the report. This notebook loads the model weights and executes analysis and visualisations on the model analysed in the report. However, some of the plots use [networkx spiral layout](https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spiral_layout.html) algorithm which is stochastic. Therefore, by running the notebook, it is possible to get slightly different plots containing the same content.
