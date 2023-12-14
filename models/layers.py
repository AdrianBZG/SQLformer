"""
Module that defines common NN layers to be used across models
"""

import math
import torch
from torch import nn
from models.utils import xavier_init_weights


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_len = max_len ** 2
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,
                                max_len,
                                dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
