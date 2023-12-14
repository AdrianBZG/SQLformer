"""
Module to wrap the dataset into a PyTorch Dataset
"""
import logging
import time
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch_geometric.transforms as T

graph_transforms = T.Compose([T.ToUndirected(),
                              T.AddSelfLoops()])

from loaders.dataset_loader import load_dataset_schemas
from loaders.dataset_loader import load_dataset_questions
from loaders.dataset_loader import load_dataset_queries
from preprocessing.vocabulary_handler import build_dataset_vocab


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("dataset")

AST_NODE_TYPES = ['ADD', 'AND', 'AVG', 'BETWEEN', 'COLUMN', 'COUNT', 'DISTINCT', 'DIV', 'EQ', 'EXCEPT', 'FROM',
                  'GROUP', 'GT', 'GTE', 'HAVING', 'IDENTIFIER', 'IN', 'INTERSECT', 'JOIN', 'LIKE', 'LIMIT',
                  'LITERAL', 'LT', 'LTE', 'MAX', 'MIN', 'NEG', 'NEQ', 'NOT', 'OR', 'ORDER', 'ORDERED', 'PAREN',
                  'SELECT', 'STAR', 'SUB', 'SUBQUERY', 'SUM', 'TABLE', 'TABLEALIAS', 'UNION', 'WHERE']

AST_NODE_TYPES = ['NONE'] + AST_NODE_TYPES


class SpiderGraphDataset(Dataset):
    """
    Subclass of PyTorch torch.utils.data.Dataset class to wrap the Spider dataset
    """

    def __init__(self, root_path="./spider", split="dev", max_prev_node=4, vocabulary=None):
        """
        :param root_path: root data path
        :param split: data split in {"train_spider", "dev", "train_others"}
        :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node default is 4
        """
        assert split in {"train_spider", "dev", "train_others"}

        logger.info(f"Loading Spider graph dataset for split {split}...")
        start_time = time.time()

        dataset_schemas_path = f"{root_path}/spider_schemas_graph.pickle"
        dataset_questions_path = f"{root_path}/{split}_questions_graph.pickle"
        dataset_queries_path = f"{root_path}/{split}_queries_graph.pickle"

        # Load schemas
        logger.info(f"Loading Spider database schemas...")
        self.schemas = load_dataset_schemas(file_path=dataset_schemas_path)
        self.db_id_to_index = dict()
        for schema in self.schemas:
            self.db_id_to_index[self.schemas[schema]["db_id"]] = schema

        # Load questions
        logger.info(f"Loading Spider graph questions...")
        self.questions = load_dataset_questions(file_path=dataset_questions_path)

        # Load queries
        logger.info(f"Loading Spider graph queries...")
        ids, x_adj, x_node_types, y_adj, y_node_types, seq_len, \
            db_ids, y_table_targets, y_column_targets = load_dataset_queries(file_path=dataset_queries_path,
                                                                                            max_prev_node=max_prev_node)

        self.ids = ids
        self.db_ids = db_ids  # To match questions and queries with the respective DB that they refer to
        self.x_adj = x_adj
        self.x_node_types = x_node_types
        self.y_adj = y_adj
        self.y_node_types = y_node_types
        self.seq_len = seq_len
        self.vocabulary = vocabulary
        self.y_table_targets = y_table_targets
        self.y_column_targets = y_column_targets

        logger.info(f"Dataset loading completed, took {round(time.time() - start_time, 2)} seconds. "
                    f"Dataset size: {len(self)}")

    def __len__(self):
        r"""
        :return: data length
        """
        return len(self.seq_len)

    def __getitem__(self, idx):
        r"""
        :param idx: index in the data
        :return: chosen data point
        """
        out_dict = dict()

        out_dict["x_adj"] = self.x_adj[idx]
        out_dict["x_node_types"] = self.x_node_types[idx]
        out_dict["y_adj"] = self.y_adj[idx]
        out_dict["y_node_types"] = self.y_node_types[idx]
        out_dict["questions"] = self.questions[idx]
        out_dict["seq_len"] = self.seq_len[idx]
        out_dict["ids"] = self.ids[idx]
        out_dict["db_ids"] = self.db_ids[idx]
        out_dict["schemas"] = self.schemas[self.db_ids[idx]]
        out_dict["vocabulary"] = self.vocabulary
        out_dict["y_table_targets"] = self.y_table_targets[idx]
        out_dict["y_column_targets"] = self.y_column_targets[idx]

        return out_dict


def get_multi_hot_targets(target_tokens, vocabulary):
    target_tokens_idx = [vocabulary.to_idx(target_token) for target_token in target_tokens]
    multi_hot = [0] * len(vocabulary)

    for idx in target_tokens_idx:
        multi_hot[idx] = 1

    return multi_hot


def generate_square_subsequent_mask(sz, device):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)


def pad_and_mask_graph(adj):
    # Sort by length descending
    seq_len = torch.IntTensor([ast.shape[0] for ast in adj])

    longest_ast_len = max(seq_len).item()

    # Mask for padding
    padder = torch.zeros(1, adj[0].shape[1]).to(adj[0].device)
    padding_mask = torch.full((len(adj), longest_ast_len),
                              False,
                              device=adj[0].device)

    for graph_idx, graph in enumerate(adj):
        graph_len = graph.shape[0]
        while graph_len != longest_ast_len:
            graph = torch.cat([graph,
                               padder],
                              dim=0)
            padding_mask[graph_idx][graph_len] = True
            graph_len += 1

        adj[graph_idx] = graph

    # Mask for future steps in attention
    nopeak_mask = generate_square_subsequent_mask(longest_ast_len,
                                                  device=adj[0].device)
    padding_mask = padding_mask

    adj = torch.stack(adj)
    return adj, nopeak_mask, padding_mask


def ordered_seq_len_collate_function(batch, tables_vocab, columns_vocab):
    r"""
    Collate function that orders the elements in a batch by descending sequence length

    :param batch: Batch from pytorch dataloader
    :return: the ordered batch
    """
    x_adj = list()
    x_node_types = list()
    y_adj = list()
    y_node_types = list()
    questions_combined_graph = list()
    questions = list()
    seq_len = list()
    ids = list()
    db_ids = list()
    schemas = list()
    queries = list()
    y_table_targets = list()
    y_column_targets = list()
    vocabulary = batch[0]["vocabulary"]

    for element in batch:
        x_adj.append(element["x_adj"])
        x_node_types.append(element["x_node_types"])
        y_adj.append(element["y_adj"])
        y_node_types.append(element["y_node_types"])
        y_table_tokens_one_hot = torch.FloatTensor(get_multi_hot_targets(element["y_table_targets"],
                                                                         tables_vocab))
        y_column_tokens_one_hot = torch.FloatTensor(get_multi_hot_targets(element["y_column_targets"],
                                                                          columns_vocab))
        y_table_targets.append(y_table_tokens_one_hot)
        y_column_targets.append(y_column_tokens_one_hot)
        questions.append(" ".join(element["questions"]["question_strip"]))

        # Apply graph transformations
        combined_graph_transformed = graph_transforms(element["questions"]["pyg"]["combined"])
        questions_combined_graph.append(combined_graph_transformed)

        seq_len.append(element["seq_len"])
        ids.append(element["ids"])
        db_ids.append(torch.tensor(element["db_ids"], dtype=torch.long))
        schemas.append(element["schemas"]["pyg"])
        queries.append(element["questions"]["query"]["query"])

    # Decoder inputs
    x_adj, x_adj_nopeak_mask, x_adj_padding_mask = pad_and_mask_graph(x_adj)
    x_node_types, x_node_types_nopeak_mask, x_node_types_padding_mask = pad_and_mask_graph(x_node_types)

    # Targets
    y_adj, _, _ = pad_and_mask_graph(y_adj)
    y_node_types, _, _ = pad_and_mask_graph(y_node_types)

    seq_len = torch.stack(seq_len)
    db_ids = torch.stack(db_ids)
    y_table_targets = torch.stack(y_table_targets)
    y_column_targets = torch.stack(y_column_targets)

    assert len(schemas) == len(db_ids)

    # Return as dictionary
    out_dict = dict()

    out_dict["x_adj"] = x_adj
    out_dict["x_adj_padding_mask"] = x_adj_padding_mask
    out_dict["x_adj_nopeak_mask"] = x_adj_nopeak_mask
    out_dict["x_node_types"] = x_node_types
    out_dict["x_node_types_padding_mask"] = x_node_types_padding_mask
    out_dict["x_node_types_nopeak_mask"] = x_node_types_nopeak_mask
    out_dict["y_adj"] = y_adj
    out_dict["y_node_types"] = y_node_types
    out_dict["y_table_targets"] = y_table_targets
    out_dict["y_column_targets"] = y_column_targets
    out_dict["questions_combined_graph"] = questions_combined_graph
    out_dict["seq_len"] = seq_len
    out_dict["ids"] = ids
    out_dict["db_ids"] = db_ids
    out_dict["schemas"] = schemas
    out_dict["queries"] = queries
    out_dict["questions"] = questions

    # For literals
    out_dict["vocabulary"] = vocabulary

    return out_dict


def get_dataset(root_path, split="dev", max_prev_node=4):
    train_spider_path = f"{root_path}/train_spider.json"
    tables_spider_path = f"{root_path}/tables.json"
    vocabulary, tables_vocab, columns_vocab = build_dataset_vocab(train_spider_path,
                                     tables_spider_path)

    dataset = SpiderGraphDataset(root_path=root_path,
                                 split=split,
                                 max_prev_node=max_prev_node,
                                 vocabulary=vocabulary)

    return dataset, vocabulary, tables_vocab, columns_vocab


def get_data_loader(dataset, tables_vocab, columns_vocab, batch_size=16, shuffle=False, num_workers=4):
    train_sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=train_sampler if shuffle else None,
                             num_workers=num_workers,
                             collate_fn=lambda batch: ordered_seq_len_collate_function(batch,
                                                                                       tables_vocab,
                                                                                       columns_vocab))

    return data_loader


if __name__ == "__main__":
    dataset = SpiderGraphDataset(root_path="data/spider",
                                 split="dev",
                                 max_prev_node=4)

    dataloader = DataLoader(dataset,
                            batch_size=16,
                            shuffle=False,
                            collate_fn=lambda batch: ordered_seq_len_collate_function(batch))

    start_time = time.time()
    logger.info(f'Dataloader size: {len(dataloader)}')
    for datapoint in dataloader:
        logger.info(f'Datapoint: {datapoint}')
        break

    logger.info(f"Iteration over the data completed, took {round(time.time() - start_time, 2)}s!")
