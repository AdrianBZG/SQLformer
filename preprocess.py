"""
Preprocessing functions to convert Spider database schemas into graphs
"""

import argparse

from config.configs import configs
from preprocessing.spider_schemas_to_graph import read_dataset_schemas
from preprocessing.spider_questions_to_graph import read_dataset_questions
from preprocessing.spider_queries_to_graph import read_dataset_queries
from preprocessing.vocabulary_handler import build_schema_vocab, build_questions_vocab
from loaders.dataset_loader import load_dataset_schemas
from data.dataset import AST_NODE_TYPES
from utils import save_to_pickle

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("preprocessing")


def _generate_schema(output_path, tables_path, schema_vocab):
    logger.info("# Generating Spider schemas as graph structure")
    spider_schema = read_dataset_schemas(tables_path,
                                         db_id_as_index=True,
                                         vocabulary=schema_vocab)

    logger.info(spider_schema.get(list(spider_schema.keys())[0]))

    spider_schema_networkx = spider_schema.get(list(spider_schema.keys())[0])["networkx"]
    logger.info(spider_schema_networkx)

    spider_schema_pyg = spider_schema.get(list(spider_schema.keys())[0])["pyg"]
    logger.info(spider_schema_pyg)

    file_path_to_save = f"{output_path}/spider_schemas_graph.pickle"
    save_to_pickle(spider_schema, file_path_to_save)


def _generate_questions(spider_path, questions_vocabulary, splits, output_path):
    logger.info("# Generating NL questions from Spider as graph structure")
    for split in splits:
        logger.info(f'## Generating for split "{split}"')

        # Read schema
        spider_questions = read_dataset_questions(spider_path[split],
                                                  vocabulary=questions_vocabulary[split])

        logger.info(spider_questions[0]["networkx"]["pos"])
        logger.info(spider_questions[0]["pyg"]["pos"])

        file_path_to_save = f"{output_path}/{split}_questions_graph.pickle"
        save_to_pickle(spider_questions, file_path_to_save)
        logger.info(f'## Saving {len(spider_questions)} questions for split "{split}"')


def _generate_queries(spider_path, root_path, tables_vocab, columns_vocab, mappings, output_path):
    logger.info("# Generating queries from Spider as BFS graph structure")

    # Getting the DB_ID -> Index mapping
    dataset_schemas_path = f"{root_path}/spider_schemas_graph.pickle"
    schemas = load_dataset_schemas(file_path=dataset_schemas_path)
    db_id_to_index = dict()
    for schema in schemas:
        db_id_to_index[schemas[schema]["db_id"]] = schema

    for split, path in spider_path.items():
        logger.info(f'## Generating for split "{split}"')

        # Read queries
        spider_queries = read_dataset_queries(path,
                                              sql_node_types_vocabulary=AST_NODE_TYPES,
                                              db_id_to_index=db_id_to_index,
                                              tables_vocab=tables_vocab,
                                              columns_vocab=columns_vocab,
                                              col_name_to_original_mapping=mappings["cols"],
                                              table_name_to_original_mapping=mappings["tables"]
                                              )

        file_path_to_save = f"{output_path}/{split}_queries_graph.pickle"
        save_to_pickle(spider_queries, file_path_to_save)
        logger.info(f'## Saving {len(spider_queries)} queries for split "{split}"')


def run_preprocessing(config):
    # Load required configs from the config dict
    root_path = config.get('root_path')
    output_path = config.get('output_path')
    tables_path = f"{root_path}/tables.json"
    spider_path = {'train_spider': f"{root_path}/train_spider.json",
                   'dev': f"{root_path}/dev.json"}
    generate_schema = config.get('generate_schema')
    generate_questions = config.get('generate_questions')
    generate_queries = config.get('generate_queries')
    splits = config.get('splits')

    # Create schemas
    schema_vocab, tables_vocab, columns_vocab, \
        col_name_to_original_mapping, table_name_to_original_mapping = build_schema_vocab(tables_path)

    questions_vocabulary = dict()
    questions_vocabulary['train_spider'] = build_questions_vocab(spider_path['train_spider'])
    questions_vocabulary['dev'] = build_questions_vocab(spider_path['dev'])

    mappings = {"cols": col_name_to_original_mapping,
                "tables": table_name_to_original_mapping}

    if generate_schema:
        _generate_schema(output_path,
                         tables_path,
                         schema_vocab)

    if generate_questions:
        _generate_questions(spider_path,
                            questions_vocabulary,
                            splits,
                            output_path)

    if generate_queries:
        _generate_queries(spider_path,
                          root_path,
                          tables_vocab,
                          columns_vocab,
                          mappings,
                          output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing pipeline')
    parser.add_argument('--config_name', type=str, nargs='?', help='The preprocessing config to use', required=True)
    args = parser.parse_args()

    if args.config_name not in configs["preprocessing"]:
        raise ValueError(f"Preprocessing config {args.config_name} does not exist")

    run_preprocessing(configs["preprocessing"][args.config_name])
