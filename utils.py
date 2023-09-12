"""
Module that contains utility functions
"""

import pickle
import torch
import torch.nn.functional as F
import torch_optimizer as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import BCEWithLogitsLoss


def masked_bce_with_logits_loss(output, target, mask):
    loss = torch.tensor([0],
                        device=target.device,
                        dtype=torch.float,
                        requires_grad=True)

    for batch_out, batch_trg, batch_mask in zip(output, target, mask):
        for seq_out, seq_trg, seq_mask in zip(batch_out, batch_trg, batch_mask):
            if seq_mask:
                continue

            loss = loss + F.binary_cross_entropy_with_logits(seq_out,
                                                             seq_trg)

    return loss


def masked_cross_entropy_loss(output, target, mask):
    loss = torch.tensor([0],
                        device=target.device,
                        dtype=torch.float,
                        requires_grad=True)

    for batch_out, batch_trg, batch_mask in zip(output, target, mask):
        for seq_out, seq_trg, seq_mask in zip(batch_out, batch_trg, batch_mask):
            if seq_mask:
                continue

            loss = loss + F.cross_entropy(seq_out,
                                          seq_trg)

    return loss


def calculate_loss(x_adj_padding_mask, y_adj, output_adj, output_node_type, y_node_types,
                   tables_logits, y_table_targets, columns_logits, y_column_targets, config):
    loss_adj = masked_bce_with_logits_loss(output_adj,
                                           y_adj,
                                           x_adj_padding_mask)

    loss_node_types = masked_cross_entropy_loss(output_node_type,
                                                y_node_types,
                                                x_adj_padding_mask)

    loss_table_concepts = masked_cross_entropy_loss(tables_logits,
                                                    y_table_targets)

    loss_column_concepts = masked_cross_entropy_loss(columns_logits,
                                                     y_column_targets)

    loss_lambda = config.get('loss_lambda')

    loss = loss_lambda[0] * loss_adj \
        + loss_lambda[1] * loss_node_types \
        + loss_lambda[2] * loss_table_concepts \
        + loss_lambda[3] * loss_column_concepts

    return loss, loss_adj, loss_node_types, loss_table_concepts, loss_column_concepts


def create_optimizer(optimizer_type, model, config):
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(list(model.parameters()),
                                      lr=config.get('learning_rate'),
                                      weight_decay=config.get('weight_decay'))
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(list(model.parameters()),
                                     lr=config.get('learning_rate'),
                                     betas=(0.9, 0.98),
                                     eps=1e-9)
    elif optimizer_type == "adafactor":
        optimizer = optim.Adafactor(model.parameters(),
                                    scale_parameter=True,
                                    relative_step=False,
                                    warmup_init=False,
                                    lr=config.get('learning_rate'),
                                    weight_decay=config.get('weight_decay'))
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def pad_and_mask_combined_graph(unbatched_graphs):
    # Sort by length descending
    seq_len = torch.IntTensor([graph.shape[0] for graph in unbatched_graphs])
    unbatched_graphs = [unbatched_graphs[i] for i in range(len(unbatched_graphs))]

    longest_graph_len = max(seq_len).item()
    padder = torch.zeros(1, unbatched_graphs[0].shape[1]).to(unbatched_graphs[0].device)
    padding_mask = torch.full((len(unbatched_graphs), longest_graph_len + 2),  # For v_table and v_column tokens
                              True,
                              device=unbatched_graphs[0].device)

    for graph_idx, graph in enumerate(unbatched_graphs):
        graph_len = graph.shape[0]
        while graph_len != longest_graph_len:
            graph = torch.cat([graph,
                               padder],
                              dim=0)
            padding_mask[graph_idx][graph_len+2] = False
            graph_len += 1

        unbatched_graphs[graph_idx] = graph

    padding_mask = padding_mask
    unbatched_graphs = torch.stack(unbatched_graphs)
    return unbatched_graphs, padding_mask, seq_len


def construct_future_mask(seq_len, device="cuda"):
    """
    Construct a binary mask that contains 1's for all valid connections and 0's for all outgoing future connections.
    This mask will be applied to the attention logits in decoder self-attention such that all logits with a 0 mask
    are set to -inf.

    :param seq_len: length of the input sequence
    :return: (seq_len,seq_len) mask
    """
    subsequent_mask = torch.triu(torch.full((seq_len, seq_len), 1, device=device), diagonal=1)
    return subsequent_mask == 0


def process_tensor(tensor, seq_len, mask=None, current_max_len=None, max_prev_node=None):
    tensor = pack_padded_sequence(tensor, seq_len, batch_first=True)
    tensor = pad_packed_sequence(tensor, batch_first=True)[0]
    if mask is not None:
        tensor *= mask[:, :current_max_len, :max_prev_node]
    if current_max_len is not None:
        tensor = tensor[:, :current_max_len]
    return tensor


def postprocess_model_output(output_adj, output_node_type, y_adj, y_node_types, seq_len, mask_sequence,
                             current_max_seq_len, max_prev_node):
    output_adj = process_tensor(output_adj, seq_len, mask_sequence, current_max_seq_len, max_prev_node)
    output_node_type = process_tensor(output_node_type, seq_len, current_max_len=current_max_seq_len)
    y_adj = process_tensor(y_adj, seq_len, mask_sequence, current_max_seq_len, max_prev_node)
    y_node_types = process_tensor(y_node_types, seq_len, current_max_len=current_max_seq_len)

    return output_adj, output_node_type, y_adj, y_node_types


def pad_tensor(tensor, max_size, dim=1):
    pad_size = max_size - tensor.shape[dim]
    padding = torch.zeros_like(tensor[0][0]).repeat(tensor.shape[0], pad_size)
    return torch.cat([tensor, padding], dim=dim)


def save_to_pickle(object_to_save, path):
    with open(path, 'wb') as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
