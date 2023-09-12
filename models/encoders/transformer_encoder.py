import unittest
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.qkv_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * self.qkv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.qkv_dim, hidden_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, x, src_padding_mask=None, encoder_hidden_states=None, future_mask=None):

        batch_size, sequence_length, hidden_dim = x.size()

        if encoder_hidden_states is None:
            q, k, v = self._self_attention_projection(x)
        else:
            q, k, v = self._cross_attention_projection(encoder_hidden_states, x)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        values, attn = self.scaled_dot_product(q, k, v, src_padding_mask, future_mask)
        values = values.reshape(batch_size, sequence_length, hidden_dim)

        output = self.o_proj(values)
        return output

    def _self_attention_projection(self, x):
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def _cross_attention_projection(self, encoder_hidden_states, decoder_hidden_states):
        batch_size, src_sequence_length, hidden_dim = encoder_hidden_states.shape
        batch_size, tgt_sequence_length, hidden_dim = decoder_hidden_states.shape

        w_q, w_kv = self.qkv_proj.weight.split([hidden_dim, 2 * hidden_dim])

        k, v = (
            F.linear(input=encoder_hidden_states, weight=w_kv)
            .reshape(batch_size, src_sequence_length, self.num_heads, 2 * self.qkv_dim)
            .chunk(2, dim=-1)
        )

        q = F.linear(input=decoder_hidden_states, weight=w_q).reshape(
            batch_size, tgt_sequence_length, self.num_heads, self.qkv_dim
        )

        return q, k, v

    def scaled_dot_product(self, q, k, v, src_padding_mask, future_mask):
        attn_logits = torch.matmul(q, torch.transpose(k, -2, -1),)

        attn_logits = attn_logits / math.sqrt(q.size()[-1])

        if src_padding_mask is not None or future_mask is not None:
            attn_logits = self.mask_logits(attn_logits, src_padding_mask, future_mask)  # type: ignore

        attention = F.softmax(attn_logits, dim=-1)

        values = torch.matmul(attention, v)
        return values, attention

    @staticmethod
    def mask_logits(logits, src_padding_mask, future_mask):
        if src_padding_mask is not None:
            masked_logits = logits.masked_fill(
                src_padding_mask[:, None, None, :] == 0, -1e9
            )
        if future_mask is not None:
            masked_logits = logits.masked_fill(future_mask == 0, -1e9)

        return masked_logits


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, ff_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(hidden_dim, ff_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, src_padding_mask):
        x = src * math.sqrt(self.hidden_dim)
        x = self.dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x, src_padding_mask=src_padding_mask)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, ff_dim, num_heads, dropout_p):
        super().__init__()
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, src_padding_mask):
        output = self.dropout1(
            self.self_mha.forward(x, src_padding_mask=src_padding_mask)
        )
        x = self.layer_norm1(x + output)

        output = self.dropout2(self.feed_forward(x))
        x = self.layer_norm2(x + output)
        return x
