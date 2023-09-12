"""
Module that abstracts functionality to instantiate a GNN encoder given a type
"""

from models.encoders.gcn_encoder import GCNEncoder
from models.encoders.gat_encoder import GATEncoder


def get_gnn_encoder(name, in_channels, out_channels=16, params=None):
    type_map = {"gcn": GCNEncoder,
                "gat": GATEncoder}

    if name not in type_map:
        raise ValueError(f'{name} is not a valid GNN encoder type')

    return type_map[name](in_channels, out_channels, params=params)
