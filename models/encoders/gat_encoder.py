"""
Module that defines a Graph Attention Network architecture
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm
from models.utils import xavier_init_weights


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, params):
        super().__init__()
        self.first_conv = GATv2Conv(in_channels, hidden_dim, params["heads"], dropout=params["dropout"])
        self.first_bn = BatchNorm(hidden_dim * params["heads"])

        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        if params["layers"] - 2 > 0:
            for _ in range(params["layers"] - 2):
                self.layers.append(GATv2Conv(hidden_dim * params["heads"],
                                             hidden_dim,
                                             heads=params["heads"],
                                             dropout=params["dropout"]))
                self.batch_norms.append(BatchNorm(hidden_dim * params["heads"]))

        self.last_conv = GATv2Conv(hidden_dim * params["heads"],
                                   hidden_dim,
                                   heads=1,
                                   concat=False,
                                   dropout=params["dropout"])
        self.last_bn = BatchNorm(hidden_dim)
        self.apply(xavier_init_weights)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.first_bn(self.first_conv(x, edge_index)))

        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = F.relu(batch_norm(layer(x, edge_index)))

        z = self.last_conv(x, edge_index)
        g = global_mean_pool(z, batch)

        return z, g
