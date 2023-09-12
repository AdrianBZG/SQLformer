"""
Module that defines a Graph Convolutional Network based encoder architecture
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, graph_level=True):
        super().__init__()
        self.graph_level = graph_level
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)

        if self.graph_level:
            x = global_mean_pool(x, batch)

        return x
