"""
Module that defines common utilities to be used across models
"""

import copy
import torch
from torch.autograd import Variable
from torch.nn import ModuleList
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
TRANSFORMER_MODEL = None
TRANSFORMER_TOKENIZER = None


def get_plm_transformer(model_name='roberta-base'):
    global TRANSFORMER_MODEL, TRANSFORMER_TOKENIZER
    if TRANSFORMER_MODEL is None:
        TRANSFORMER_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        TRANSFORMER_MODEL = AutoModel.from_pretrained(model_name)

    return {"model": TRANSFORMER_MODEL,
            "tokenizer": TRANSFORMER_TOKENIZER}


def adjust_softmax(y):
    max_idx = torch.argmax(y, dim=2, keepdim=True)
    y.zero_()
    one_value = torch.full(max_idx.shape, 1, dtype=torch.float).cuda()
    y.scatter_(dim=2, index=max_idx, src=one_value)
    return y


def sample_sigmoid(y, device, threshold=0.5):
    r"""
    Sample from scores between 0 and 1 as means of Bernouolli distribution, or threshold over 0.5

    :param args: parsed arguments
    :param y: values to threshold
    :param sample: if True, sample, otherwise, threshold
    :return: sampled/thresholed values, in {0., 1.}
    """
    y_thresh = (torch.ones(y.size(0), y.size(1), y.size(2)) * threshold).to(device)
    y_result = torch.gt(y, y_thresh).float()
    return y_result


def create_list_of_modules(module, number_modules):
    return ModuleList([copy.deepcopy(module) for _ in range(number_modules)])


def generate_input_sequence(x_adj, x_node_types, question_schema_ca, joint_tables_columns_embedding):
    # For the x_adj we have B x SEQ_LEN x MAX_BFS_PREV_NODES as dimension
    # So we want to add the SEQ_LEN dimension to the other embeddings
    #schema_embedding = schema_embedding.unsqueeze(1).repeat(1, x_adj.shape[1], 1)
    #question_graph_embedding = question_graph_embedding.unsqueeze(1).repeat(1, x_adj.shape[1], 1)
    question_schema_ca = question_schema_ca.unsqueeze(1).repeat(1, x_adj.shape[1], 1)
    joint_tables_columns_embedding = joint_tables_columns_embedding.unsqueeze(1).repeat(1, x_adj.shape[1], 1)

    # Now we can concatenate into a single tensor by dimension 2, resulting tensor has dimensionality of:
    # B x SEQ_LEN x (PRECOMPUTED_NODE_EMBEDDING_SIZE + GNN_ENCODERS_EMBEDDING_DIMENSIONALITY*2 + GNN_ENCODERS_EMBEDDING_DIMENSIONALITY)
    #input_sequence = torch.cat([x_adj, x_node_types, schema_embedding, question_graph_embedding, question_embedding], dim=2)
    input_sequence = torch.cat([x_adj, x_node_types, question_schema_ca, joint_tables_columns_embedding],
                               dim=2)

    return input_sequence


def xavier_init_weights(m):
    r"""
    Apply xavier initialization on Linear layers

    :param m: PyTorch layer
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
