"""
Preprocessing functions to convert Spider SQL queries into graphs
"""

import json
import random
import logging
from tqdm import tqdm
import sqlglot as sqlglot

from loaders.dataset_loader import load_dataset_schemas
from data.dataset import AST_NODE_TYPES
from preprocessing.spider_questions_to_graph import SpiderNLGraphNode, SpiderNLGraphEdge

logger = logging.getLogger("SpiderQueriesToGraph")

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class SpiderQueryGraph:
    def __init__(self,
                 identifier,
                 db_id,
                 db_id_original,
                 query,
                 root_node,
                 adjacency_list,
                 node_type_list,
                 table_tokens,
                 column_tokens,
                 vocabulary):

        self.identifier = identifier
        self.db_id = db_id
        self.db_id_original = db_id_original
        self.query = query
        self.root_node = root_node
        self.adjacency_list = adjacency_list
        self.node_type_list = node_type_list
        self.vocabulary = vocabulary
        self.table_tokens = table_tokens
        self.column_tokens = column_tokens

    def __str__(self):
        return f'([ID: {self.identifier}] DB: {self.db_id}) {self.query} | {self.adjacency_list} | {self.node_type_list}'

    def __repr__(self):
        return f'([ID: {self.identifier}] DB: {self.db_id}) {self.query} | {self.adjacency_list} | {self.node_type_list}'

    def to_adjacency_matrix(self):
        graph = self.adjacency_list
        keys = sorted(graph.keys())
        size = len(keys)

        matrix = [[0] * size for _ in range(size)]

        # We iterate over the key:value entries in the dictionary first,
        # then we iterate over the elements within the value
        for a, b in [(keys.index(a), keys.index(b)) for a, row in graph.items() for b in row]:
            # Use 1 to represent if there's an edge
            # Use 2 to represent when node meets itself in the matrix (A -> A)
            matrix[a][b] = 2 if (a == b) else 1

        return matrix

    def to_bfs_adjacency_matrix(self):
        graph = self.adjacency_list
        keys = sorted(graph.keys())
        size = len(keys)

        matrix = [[0] * size for _ in range(size)]

        for node_id in keys:
            if node_id < 1:
                # Root node doesn't have any previous nodes, so adjacency vector is all-zero
                continue

            # Check single connections as it's a tree
            if len(graph[node_id]) > 1:
                raise ValueError(f'Node ID {node_id} has more than 1 parent, which is invalid for a tree structure.')

            connected_node = graph[node_id][0]
            adjacency_vector_index = node_id - connected_node - 1
            matrix[node_id][adjacency_vector_index] = 1

        return matrix

    def get_node_type_matrix(self):
        node_type_list = self.node_type_list
        keys = sorted(node_type_list.keys())
        num_nodes = len(keys)
        vocab_size = len(self.vocabulary)
        matrix = [[0] * vocab_size for _ in range(num_nodes)]

        for node_id in keys:
            node_type = node_type_list[node_id]
            vector_index = self.vocabulary.index(node_type)
            matrix[node_id][vector_index] = 1

        return matrix

    def to_dict(self):
        item_dict = dict()

        item_dict["identifier"] = self.identifier
        item_dict["db_id"] = self.db_id
        item_dict["db_id_original"] = self.db_id_original
        item_dict["query"] = self.query
        item_dict["root_node"] = self.root_node
        item_dict["adjacency_list"] = self.adjacency_list
        item_dict["node_type_list"] = self.node_type_list
        item_dict["vocabulary"] = self.vocabulary
        item_dict["adjacency_matrix"] = self.to_adjacency_matrix()
        item_dict["bfs_adjacency_matrix"] = self.to_bfs_adjacency_matrix()
        item_dict["node_type_matrix"] = self.get_node_type_matrix()
        item_dict["table_tokens"] = self.table_tokens
        item_dict["column_tokens"] = self.column_tokens

        return item_dict


def _get_dependency_graph(stanza_input):
    dependency_graph_edges = list()
    dependencies = stanza_input.sentences[0].dependencies
    for dependency in dependencies:
        dependency_source = dependency[0]
        dependency_type = dependency[1]
        dependency_target = dependency[2]

        # Skip dependencies involving the root (id = 0)
        if dependency_source.id == 0 or dependency_target.id == 0:
            continue

        # Skip dependencies involving PUNCT elements
        if dependency_source.upos == "PUNCT" or dependency_target.upos == "PUNCT":
            continue

        # Clean the dependency type
        dependency_type = dependency_type.split(":")[0]

        logger.debug(f'{dependency_source.text} - {dependency_type} - {dependency_target.text}')

        source_word_node = SpiderNLGraphNode(dependency_source.text, "word")
        target_word_node = SpiderNLGraphNode(dependency_target.text, "word")
        dependency_edge = SpiderNLGraphEdge(source_word_node, dependency_type, target_word_node)
        dependency_graph_edges.append(dependency_edge)

    return dependency_graph_edges


def get_ast_from_query(identifier, db_id, db_id_original, query, sql_node_types_vocabulary,
                       tables_vocab, columns_vocab, col_name_to_original_mapping, table_name_to_original_mapping,
                       exclude_children=True):

    transpiled_query = sqlglot.transpile(query, write="sqlite", identify=True)[0]
    parsed_query = sqlglot.parse_one(transpiled_query)

    objectIDtoNodeID = dict()
    root_node_id = None
    adjacency_list = dict()
    node_types = dict()

    table_tokens = set()
    column_tokens = set()

    # Perform a BFS walk over the query
    for idx, (item, parent, key) in enumerate(parsed_query.walk(bfs=True)):
        if item.key.upper() == "IDENTIFIER":
            continue

        if item.key.upper() == "LITERAL" and not str(item).isdigit():
            continue

        if id(item) not in objectIDtoNodeID:
            objectIDtoNodeID[id(item)] = len(objectIDtoNodeID)
            adjacency_list[objectIDtoNodeID[id(item)]] = list()

        nodeID = objectIDtoNodeID[id(item)]

        if nodeID not in node_types:
            node_types[nodeID] = item.key.upper()

        if parent is None:
            # This is the parent Node (SELECT/UNION/etc expression)
            root_node_id = nodeID
            logger.debug(f'Item: {item.key.upper()} (ID: {id(item)}) | Parent: None | Key: {key}')

        else:
            # Get literals for each node for IDENTIFIER node types depending on the parent
            literal_identifier = None
            item_name = str(item).rstrip().strip()

            if item.key.upper() == "TABLE":
                table_name = str(item_name.replace('"', '').replace("'", "")).lower()
                table_name_alias_removal = table_name.split(" ")
                if len(table_name_alias_removal) > 1:
                    table_name = table_name_alias_removal[0]

                if not tables_vocab.is_in_vocab(table_name):
                    if table_name in table_name_to_original_mapping:
                        table_name = table_name_to_original_mapping[table_name]
                        if not tables_vocab.is_in_vocab(table_name):
                            raise ValueError(f"Table {table_name} is not in the schema vocabulary")
                    else:
                        raise ValueError(f"Table {table_name} is not in the schema vocabulary")

                if not tables_vocab.is_in_vocab(table_name):
                    raise ValueError(f"Table {table_name} is not in the tables vocabulary")

                table_tokens.add(table_name)
                literal_identifier = table_name
            elif item.key.upper() == "COLUMN":
                column_name = str(item_name.replace('"', '').replace("'", "")).lower()
                column_name_alias_removal = column_name.split(".")
                if len(column_name_alias_removal) > 1:
                    column_name = column_name_alias_removal[1]

                if not columns_vocab.is_in_vocab(column_name):
                    if column_name in col_name_to_original_mapping:
                        if columns_vocab.is_in_vocab(col_name_to_original_mapping[column_name]):
                            column_name = col_name_to_original_mapping[column_name]
                            literal_identifier = column_name
                            column_tokens.add(column_name)
                        else:
                            raise ValueError(f"Column {column_name} is not in the columns vocabulary")
                    else:
                        literal_identifier = "@val"
                else:
                    literal_identifier = column_name

                if literal_identifier != "@val":
                    column_tokens.add(literal_identifier)
            elif item.key.upper() == "LITERAL":
                literal_identifier = "@val"
            elif item.key.upper() == "STAR":
                column_tokens.add("*")

            logger.debug(f'Item: {item.key.upper()} (ID: {id(item)}) | Parent: {parent.key.upper()} | '
                         f'Parent ID: {id(parent)} | Key: {key}')
            parentNodeID = objectIDtoNodeID[id(parent)]

            if not exclude_children:
                # Exclude parents from being connected to its children. Assuming the graph is undirected anyway,
                # and making it easier for transforming into a BFS adjacency matrix
                adjacency_list[parentNodeID].append(nodeID)
            adjacency_list[nodeID].append(parentNodeID)

            if literal_identifier:
                literal_node_id = id(literal_identifier)
                if literal_identifier == "@val":
                    literal_node_id += idx
                    node_types[nodeID] = "LITERAL"

                if literal_node_id not in objectIDtoNodeID:
                    objectIDtoNodeID[literal_node_id] = len(objectIDtoNodeID)
                    adjacency_list[objectIDtoNodeID[literal_node_id]] = list()

                adjacency_list[objectIDtoNodeID[literal_node_id]].append(nodeID)
                node_types[objectIDtoNodeID[literal_node_id]] = literal_identifier

    # Create the graph object
    joint_vocab = sql_node_types_vocabulary + tables_vocab.as_list() + columns_vocab.as_list() + ["@val"]
    query_graph = SpiderQueryGraph(identifier=identifier,
                                   db_id=db_id,
                                   db_id_original=db_id_original,
                                   query=query,
                                   root_node=root_node_id,
                                   adjacency_list=adjacency_list,
                                   node_type_list=node_types,
                                   table_tokens=table_tokens,
                                   column_tokens=column_tokens,
                                   vocabulary=joint_vocab)

    logger.debug(query_graph)
    return query_graph


def get_ast_node_types(query):
    transpiled_query = sqlglot.transpile(query, write="sqlite", identify=True)[0]
    parsed_query = sqlglot.parse_one(transpiled_query)

    ast_node_types = list()
    keys = list(parsed_query.args.keys())
    logger.info(keys)
    logger.info(repr(parsed_query))
    skipped_query_parts = ["distinct", "limit"]
    for query_part in keys:
        if query_part in skipped_query_parts:
            continue

        logger.debug(query_part.upper())
        parsed_query_key = parsed_query.args[query_part]

        logger.debug(repr(parsed_query_key))

        if query_part in ("expressions", "joins"):
            if query_part == "expressions":
                # Parse SELECT statement
                if len(parsed_query_key) > 0:
                    logger.warning(f'Len: {len(parsed_query_key)} - {parsed_query_key}')
                    for selectStatement in parsed_query_key:
                        for item, parent, key in selectStatement.walk(bfs=True):
                            if item.key.upper() not in ast_node_types:
                                ast_node_types.append(item.key.upper())
            elif query_part == "joins":
                # Parse JOIN statements
                if len(parsed_query_key) > 0:
                    logger.warning(f'Len: {len(parsed_query_key)} - {parsed_query_key}')
                    for joinStatement in parsed_query_key:
                        for item, parent, key in joinStatement.walk(bfs=True):
                            if item.key.upper() not in ast_node_types:
                                ast_node_types.append(item.key.upper())
        else:
            if parsed_query_key:
                for item, parent, key in parsed_query_key.walk(bfs=True):
                    if item.key.upper() not in ast_node_types:
                        ast_node_types.append(item.key.upper())

    return ast_node_types


def read_dataset_queries(input_file_path, sql_node_types_vocabulary,
                         tables_vocab, columns_vocab, col_name_to_original_mapping,
                         table_name_to_original_mapping, db_id_to_index):
    queries = dict()
    input_file = json.load(open(input_file_path, "r"))

    for index, entry in enumerate(tqdm(input_file)):
        query_identifier = index
        db_id = db_id_to_index[entry["db_id"]]
        db_id_original = entry["db_id"]
        query = {"query": entry["query"],
                 "query_toks": entry["query_toks"],
                 "query_toks_no_value": entry["query_toks_no_value"]}

        # Transform the query into an Abstract Syntax Tree
        query_ast = get_ast_from_query(identifier=query_identifier,
                                       db_id=db_id,
                                       db_id_original=db_id_original,
                                       query=query["query"],
                                       sql_node_types_vocabulary=sql_node_types_vocabulary,
                                       tables_vocab=tables_vocab,
                                       columns_vocab=columns_vocab,
                                       col_name_to_original_mapping=col_name_to_original_mapping,
                                       table_name_to_original_mapping=table_name_to_original_mapping,
                                       )

        queries[query_identifier] = query_ast.to_dict()

    return queries


def read_dataset_node_types(input_file_path):
    input_file = json.load(open(input_file_path, "r"))
    ast_node_types = set()
    for index, entry in enumerate(tqdm(input_file)):
        # Get the query
        query = {"query": entry["query"],
                 "query_toks": entry["query_toks"],
                 "query_toks_no_value": entry["query_toks_no_value"]}

        # Get the node types
        node_types = get_ast_node_types(query["query"])
        ast_node_types.update(node_types)

    return list(ast_node_types)


if __name__ == '__main__':
    splits = ["train_spider", "dev"]

    # Getting the different AST node types
    GET_AST_NODE_TYPES = False
    if GET_AST_NODE_TYPES:
        splits = ["train_spider", "dev", "train_others"]
        ast_node_types = list()
        for split in splits:
            logger.info(f'## Generating for split "{split}"')
            split_spider_file_path = f'../data/spider/{split}.json'

            # Read queries
            spider_ast_node_types = read_dataset_node_types(split_spider_file_path)
            ast_node_types.extend(spider_ast_node_types)

        ast_node_types = sorted(list(set(ast_node_types)))
    else:
        ast_node_types = sorted(AST_NODE_TYPES)

    # Getting the DB_ID -> Index mapping
    root_path = "data/spider"
    dataset_schemas_path = f"{root_path}/spider_schemas_graph.pickle"
    schemas = load_dataset_schemas(file_path=dataset_schemas_path)
    db_id_to_index = dict()
    for schema in schemas:
        db_id_to_index[schemas[schema]["db_id"]] = schema

    # Getting the queries as AST
    splits = ["train_spider", "dev", "train_others"]
    queries_as_ast = dict()
    for split in splits:
        logger.info(f'## Generating for split "{split}"')
        split_spider_file_path = f'../data/spider/{split}.json'

        # Read queries
        queries_ast = read_dataset_queries(split_spider_file_path,
                                           sql_node_types_vocabulary=ast_node_types,
                                           db_id_to_index=db_id_to_index)

        # Add to the list of queries
        queries_as_ast[split] = queries_ast

    # Display all queries if during DEBUG mode
    for query in queries_as_ast[splits[0]].values():
        logger.debug(query)
        logger.debug(query["bfs_adjacency_matrix"])
        logger.debug(query["node_type_matrix"])

    # Display a random query
    logger.info("Random Query:")
    logger.info(queries_as_ast[splits[0]][random.randint(0, len(queries_as_ast[splits[0]]) - 1)])
    logger.info(queries_as_ast[splits[0]][random.randint(0, len(queries_as_ast[splits[0]]) - 1)]["bfs_adjacency_matrix"])
    logger.info(queries_as_ast[splits[0]][random.randint(0, len(queries_as_ast[splits[0]]) - 1)]["node_type_matrix"])
    logger.info(f'Total amount of queries: {len(queries_as_ast[splits[0]])}')
