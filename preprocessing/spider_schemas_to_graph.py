"""
Preprocessing functions to convert Spider database schemas into graphs
"""

import json
import networkx as nx
from typing import List
import logging
from tqdm import tqdm
from torch_geometric.utils.convert import from_networkx

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("SpiderSchemasToGraph")


class SpiderSchemaNode:
    def __init__(self,
                 name: str,
                 type: str):
        self.name = name
        self.type = type

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return f'{self.name}'


class SpiderSchemaEdge:
    def __init__(self,
                 source: SpiderSchemaNode,
                 edge_type: str,
                 target: SpiderSchemaNode):
        self.source = source
        self.edge_type = edge_type
        self.target = target

    def __str__(self):
        return f'({self.source.name},{self.edge_type},{self.target.name})'

    def __repr__(self):
        return f'({self.source.name},{self.edge_type},{self.target.name})'


class SpiderSchema:
    def __init__(self,
                 identifier: str,
                 graph_edge_list: List[SpiderSchemaEdge],
                 vocabulary):
        self.identifier = identifier
        self.graph = graph_edge_list
        self.vocabulary = vocabulary

    def __str__(self):
        return f'{self.identifier}: {self.graph}'

    def __repr__(self):
        return f'{self.identifier}: {self.graph}'

    def to_networkx(self):
        '''
        Transforms graph into a NetworkX Levi Graph
        '''

        schema_graph = nx.Graph()

        # Loop over relations
        for edge in self.graph:
            source_node = edge.source
            target_node = edge.target

            # Add nodes if not exist
            if source_node.name not in schema_graph:
                # Calculate node embedding using pre-trained sentence model
                node_embedding = self.vocabulary.get_word_embedding(source_node.name)
                schema_graph.add_node(source_node.name,
                                      x=node_embedding)

            if target_node.name not in schema_graph:
                node_embedding = self.vocabulary.get_word_embedding(target_node.name)
                schema_graph.add_node(target_node.name,
                                      x=node_embedding)

            if edge.edge_type not in schema_graph:
                node_embedding = self.vocabulary.get_word_embedding(edge.edge_type)

                schema_graph.add_node(edge.edge_type,
                                      x=node_embedding)

            # Add edges
            schema_graph.add_edge(source_node.name, edge.edge_type)
            schema_graph.add_edge(target_node.name, edge.edge_type)

        undirected_schema_graph = schema_graph.to_undirected()
        return undirected_schema_graph

    def to_pyg(self):
        # Obtain the NetworkX representation
        networkx_graph = self.to_networkx()

        # Convert the graph into PyTorch geometric
        pyg_graph = from_networkx(networkx_graph)

        logger.info(pyg_graph)
        logger.info(pyg_graph.edge_index)

        return pyg_graph

    def to_dict(self):
        item_dict = dict()

        item_dict["db_id"] = self.identifier
        item_dict["pyg"] = self.to_pyg()
        item_dict["networkx"] = self.to_networkx()

        return item_dict


def read_dataset_schemas(schema_path, vocabulary, db_id_as_index=True):
    schemas = dict()
    schemas_json_blob = json.load(open(schema_path, "r"))

    for index, db in enumerate(tqdm(schemas_json_blob)):
        # Make the networkx graph
        schema_edges = list()

        db_id = db['db_id']

        column_id_to_table = {}
        column_id_to_column = {}
        schema_node_map = dict()

        for i, (column, text, column_type) in enumerate(zip(db['column_names_original'],
                                                            db['column_names'],
                                                            db['column_types'])):
            table_id, column_name = column
            _, column_text = text

            table_name = db['table_names_original'][table_id]

            if column_name == "*":
                continue

            # Add edge for table-column
            table_node = SpiderSchemaNode(table_name, "table")
            column_node = SpiderSchemaNode(table_name + ":" + column_name, "column")
            table_column_edge = SpiderSchemaEdge(table_node, "has_column", column_node)
            schema_edges.append(table_column_edge)

            schema_node_map[table_node.name] = table_node
            schema_node_map[column_node.name] = column_node

            # Add edge for column type
            column_type_node = SpiderSchemaNode(column_type, "column_type")
            schema_node_map[column_type_node.name] = column_type_node
            column_type_edge = SpiderSchemaEdge(column_node, "is_type", column_type_node)
            schema_edges.append(column_type_edge)

            # Add to lookup map
            column_id_to_table[i] = table_name
            column_id_to_column[i] = column_name

        # Foreign keys
        for (c1, c2) in db['foreign_keys']:
            fk_column_name_source = column_id_to_table[c1] + ':' + column_id_to_column[c1]
            fk_column_name_target = column_id_to_table[c2] + ':' + column_id_to_column[c2]

            if fk_column_name_source not in schema_node_map:
                logger.warning(f'{fk_column_name_source} was not in the schema_node_map. Skipping FK relation')
                continue

            if fk_column_name_target not in schema_node_map:
                logger.warning(f'{fk_column_name_target} was not in the schema_node_map. Skipping FK relation')
                continue

            fk_column_source_node = schema_node_map[fk_column_name_source]
            fk_column_target_node = schema_node_map[fk_column_name_target]
            foreign_key_edge = SpiderSchemaEdge(fk_column_source_node, "is_foreign_key", fk_column_target_node)
            schema_edges.append(foreign_key_edge)

        schema = SpiderSchema(db_id,
                              schema_edges,
                              vocabulary=vocabulary)
        if db_id_as_index:
            schemas[index] = schema.to_dict()  # Save as dict to unpickle
        else:
            schemas[db_id] = schema.to_dict()  # Save as dict to unpickle

    return {**schemas}
