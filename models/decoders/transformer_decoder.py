import math
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_p,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout_p)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )

    def _reset_parameters(self):
        """ Perform xavier weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self,
        encoder_hidden_states,
        x_adj,
        x_node_types,
        src_padding_mask=None,
        future_mask=None
    ):
        x_adj = x_adj * math.sqrt(self.hidden_dim)
        x_adj = self.dropout(x_adj)

        x_node_types = x_node_types * math.sqrt(self.hidden_dim)
        x_node_types = self.dropout(x_node_types)

        for decoder_block in self.decoder_blocks:
            x_adj, x_nt = decoder_block(encoder_hidden_states=encoder_hidden_states,
                                        src_padding_mask=src_padding_mask,
                                        x_adj=x_adj,
                                        future_mask=future_mask,
                                        x_node_types=x_node_types)

        return x_adj, x_nt


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.qkv_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * self.qkv_dim, bias=False)
        self.e_proj = nn.Linear(hidden_dim, num_heads * self.qkv_dim, bias=False)

        self.o_proj_1 = nn.Linear(num_heads * self.qkv_dim, hidden_dim, bias=False)
        self.o_proj_2 = nn.Linear(num_heads * self.qkv_dim, hidden_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.e_proj.weight)
        nn.init.xavier_uniform_(self.o_proj_1.weight)
        nn.init.xavier_uniform_(self.o_proj_2.weight)

    def forward(
        self,
        x_adj,
        x_nt,
        src_padding_mask=None,
        encoder_hidden_states=None,
        future_mask=None,
    ):
        batch_size, sequence_length, hidden_dim = x_adj.size()

        if encoder_hidden_states is None:
            q, k, v, e = self._self_attention_projection(x_adj, x_nt)
        else:
            q, k, v, e = self._cross_attention_projection(encoder_hidden_states, x_adj, x_nt)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        e = e.permute(0, 2, 1, 3)

        values, attn = self.scaled_dot_product(q, k, v, e, src_padding_mask, future_mask)
        values = values.reshape(batch_size, sequence_length, hidden_dim)

        output_1 = self.o_proj_1(values)
        output_2 = self.o_proj_2(values)
        return output_1, output_2

    def _self_attention_projection(self, x_adj, x_nt):
        # ADJ
        batch_size, sequence_length, _ = x_adj.shape
        qkv = self.qkv_proj(x_adj)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # NT
        batch_size_nt, sequence_length_nt, _ = x_nt.shape
        e = self.e_proj(x_nt)
        e = e.reshape(batch_size, sequence_length, self.num_heads, self.qkv_dim)

        return q, k, v, e

    def _cross_attention_projection(self, encoder_hidden_states, x_adj, x_nt):
        batch_size, tgt_sequence_length, hidden_dim = x_adj.shape

        w_q, w_kv = self.qkv_proj.weight.split([hidden_dim, 2 * hidden_dim])

        k, v = (
            F.linear(input=encoder_hidden_states, weight=w_kv)
            .reshape(batch_size, encoder_hidden_states.shape[1], self.num_heads, 2 * self.qkv_dim)
            .chunk(2, dim=-1)
        )

        q = F.linear(input=x_adj, weight=w_q).reshape(
            batch_size, tgt_sequence_length, self.num_heads, self.qkv_dim
        )

        # Project node types into e's
        batch_size_nt, sequence_length_nt, _ = x_nt.shape
        e = self.e_proj(x_nt)
        e = e.reshape(batch_size, tgt_sequence_length, self.num_heads, self.qkv_dim)

        return q, k, v, e

    def scaled_dot_product(self, q, k, v, e, src_padding_mask, future_mask):
        attn_logits = torch.matmul(q, torch.transpose(k, -2, -1),)

        attn_logits = attn_logits / math.sqrt(q.size()[-1])

        if src_padding_mask is not None or future_mask is not None:
            attn_logits = self.mask_logits(attn_logits, src_padding_mask, future_mask)  # type: ignore

        # Include node type 'e' projection into the values
        e = e / math.sqrt(q.size()[-1])
        values = torch.einsum('bijk,blmn->blmn', e, v)

        attention = F.softmax(attn_logits, dim=-1)

        values = torch.matmul(attention, values)

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


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()

        self.self_mha = CustomMultiHeadAttention(hidden_dim, num_heads)
        self.cross_mha = CustomMultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_hidden_states, src_padding_mask, x_adj, x_node_types, future_mask):
        o_proj_1, o_proj_2 = self.self_mha.forward(x_adj=x_adj,
                                                   future_mask=future_mask,
                                                   x_nt=x_node_types)

        output_1 = self.dropout1(o_proj_1)
        output_2 = self.dropout1(o_proj_2)

        x_adj = self.layer_norm1(x_adj + output_1)
        x_nt = self.layer_norm1(x_node_types + output_2)

        o_proj_1, o_proj_2 = self.cross_mha.forward(
                                x_adj=x_adj,
                                x_nt=x_nt,
                                encoder_hidden_states=encoder_hidden_states,
                                src_padding_mask=src_padding_mask,
                            )
        output_1 = self.dropout2(o_proj_1)
        output_2 = self.dropout2(o_proj_2)

        x_adj = self.layer_norm2(x_adj + output_1)
        x_nt = self.layer_norm2(x_nt + output_2)

        # Feed forward layers (ADJ)
        output_adj = self.dropout3(self.feed_forward(x_adj))
        x_adj = self.layer_norm3(x_adj + output_adj)

        # Feed forward layers (NT)
        output_nt = self.dropout3(self.feed_forward(x_nt))
        x_nt = self.layer_norm3(x_nt + output_nt)

        return x_adj, x_nt
