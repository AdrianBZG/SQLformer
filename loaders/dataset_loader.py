"""
Helper methods to load the dataset in a format usable by downstream scripts
"""

import pickle
import logging
import torch
import os

from utils import pad_tensor

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("dataset_loader")


def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} was not found')

    return True


def load_dataset_schemas(file_path):
    check_file_exists(file_path)

    with open(file_path, 'rb') as handle:
        logger.info(file_path)
        schemas = pickle.load(handle)

    return schemas


def load_dataset_questions(file_path):
    check_file_exists(file_path)

    with open(file_path, 'rb') as handle:
        questions = pickle.load(handle)

    return questions


def load_dataset_queries(file_path, max_prev_node):
    check_file_exists(file_path)

    with open(file_path, 'rb') as handle:
        queries = pickle.load(handle)

    ids = list(queries.keys())

    list_x_adj, list_x_node_types, list_y_adj, list_y_node_types = [], [], [], []
    list_seq_len, list_db_ids, list_y_table_targets, list_y_column_targets = [], [], [], []

    for id in ids:
        query = queries[id]
        x_adj = torch.FloatTensor(query["bfs_adjacency_matrix"])
        x_node_types = torch.FloatTensor(query["node_type_matrix"])
        query_db_id = query["db_id"]
        y_table_tokens, y_column_tokens = query["table_tokens"], query["column_tokens"]

        pad_node_type = torch.zeros_like(x_node_types[0])
        pad_node_type[0] = 1.

        y_adj = torch.cat([x_adj[1:, :].clone(), torch.zeros_like(x_adj[0]).unsqueeze(0)])
        y_node_types = torch.cat([x_node_types[1:, :].clone(), pad_node_type.unsqueeze(0)])

        x_adj, y_adj = x_adj[:, :max_prev_node], y_adj[:, :max_prev_node]
        if x_adj.shape[1] < max_prev_node:
            x_adj = pad_tensor(x_adj, max_prev_node)
        if y_adj.shape[1] < max_prev_node:
            y_adj = pad_tensor(y_adj, max_prev_node)

        list_x_adj.append(x_adj)
        list_x_node_types.append(x_node_types)
        list_y_adj.append(y_adj)
        list_y_node_types.append(y_node_types)
        list_seq_len.append(len(x_adj))
        list_db_ids.append(query_db_id)
        list_y_table_targets.append(y_table_tokens)
        list_y_column_targets.append(y_column_tokens)

    list_seq_len = torch.LongTensor(list_seq_len)

    return ids, list_x_adj, list_x_node_types, list_y_adj, list_y_node_types, list_seq_len, list_db_ids, \
           list_y_table_targets, list_y_column_targets
