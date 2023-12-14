"""
Preprocessing functions to convert Spider database schemas into graphs
"""

import json
import logging
from tqdm import tqdm
import torch

from utils import get_sentence_embedding
from preprocessing.structures import SpiderSchemaGraph, SpiderInputGraphNode, SpiderInputGraphEdge

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("SpiderSchemasToGraph")


def read_dataset_schemas(schema_path, vocabulary, db_id_as_index=True):
    schemas = dict()
    schemas_json_blob = json.load(open(schema_path, "r"))

    for index, db in enumerate(tqdm(schemas_json_blob)):
        # Make the networkx graph
        schema_edges = list()

        db_id = db['db_id']

        column_id_to_table = {}
        column_id_to_column = {}
        schema_node_map = {}
        table_embeddings = {}
        column_embeddings = {}

        for i, (column, text, column_type) in enumerate(zip(db['column_names_original'],
                                                            db['column_names'],
                                                            db['column_types'])):
            table_id, column_name = column
            _, column_text = text

            table_name = db['table_names_original'][table_id]

            if column_name == "*":
                continue

            # Get sentence embedding for the table
            if table_name not in table_embeddings:
                table_embedding = get_sentence_embedding(table_name)
                table_embedding = torch.stack([emb for emb in table_embedding.values()]).mean(dim=0)
                table_embeddings[table_name] = table_embedding

            if column_name not in column_embeddings:
                column_embedding = get_sentence_embedding(column_name)
                column_embedding = torch.stack([emb for emb in column_embedding.values()]).mean(dim=0)
                column_embeddings[column_name] = column_embedding

            # Add edge for table-column
            table_node = SpiderInputGraphNode(table_name, "table",
                                              embedding=table_embeddings[table_name])
            column_node = SpiderInputGraphNode(table_name + ":" + column_name, "column",
                                               embedding=column_embeddings[column_name])
            table_column_edge = SpiderInputGraphEdge(table_node, "has_column", column_node)
            schema_edges.append(table_column_edge)

            schema_node_map[table_node.name] = table_node
            schema_node_map[column_node.name] = column_node

            # Add edge for column type
            column_type_node = SpiderInputGraphNode(column_type, "column_type",
                                                    embedding=torch.randn_like(table_embeddings[table_name]))
            schema_node_map[column_type_node.name] = column_type_node
            column_type_edge = SpiderInputGraphEdge(column_node, "is_type", column_type_node)
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
            foreign_key_edge = SpiderInputGraphEdge(fk_column_source_node, "is_foreign_key", fk_column_target_node)
            schema_edges.append(foreign_key_edge)

        schema = SpiderSchemaGraph(db_id,
                                   schema_edges,
                                   vocabulary=vocabulary)
        if db_id_as_index:
            schemas[index] = schema.to_dict()  # Save as dict to unpickle
        else:
            schemas[db_id] = schema.to_dict()  # Save as dict to unpickle

    return {**schemas}
