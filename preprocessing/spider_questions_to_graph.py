"""
Preprocessing functions to convert Spider questions into graphs
"""

import json
import networkx as nx
from typing import List
import logging
from stanza import DownloadMethod
from tqdm import tqdm
import torch
from torch_geometric.utils.convert import from_networkx
import stanza

from utils import get_sentence_embedding
from preprocessing.structures import SpiderQuestionGraph, SpiderInputGraphNode, SpiderInputGraphEdge
from preprocessing.vocabulary_handler import strip_nl

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("SpiderQuestionsToGraph")


def _get_pos_tag_graph(stanza_input, sentence_embeddings):
    pos_graph_edges = list()
    pos_tagging = dict()
    for sentence in stanza_input.sentences:
        for word in sentence.words:
            # Skip PUNCT tokens (e.g. ?, ., etc)
            if word.pos in ["PUNCT"]:
                continue

            pos_tagging[word.text] = word.pos
            logger.debug(f'{word.text}: {word.pos}')

    for word, pos in pos_tagging.items():
        word_node = SpiderInputGraphNode(word, "word",
                                         embedding=sentence_embeddings[word])
        pos_tag_node = SpiderInputGraphNode(pos, "pos_tag",
                                            embedding=torch.randn_like(sentence_embeddings[word]))
        word_pos_edge = SpiderInputGraphEdge(word_node, "has_pos", pos_tag_node)
        pos_graph_edges.append(word_pos_edge)

    return pos_graph_edges


def _get_dependency_graph(stanza_input, sentence_embeddings):
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

        source_word_node = SpiderInputGraphNode(dependency_source.text, "word",
                                                embedding=sentence_embeddings[dependency_source.text])
        target_word_node = SpiderInputGraphNode(dependency_target.text, "word",
                                                embedding=sentence_embeddings[dependency_target.text])
        dependency_edge = SpiderInputGraphEdge(source_word_node, dependency_type, target_word_node)
        dependency_graph_edges.append(dependency_edge)

    return dependency_graph_edges


def read_dataset_questions(input_file_path, vocabulary):
    questions = dict()
    input_file = json.load(open(input_file_path, "r"))
    stanza_nlp = stanza.Pipeline('en',
                                 processors='depparse,pos,tokenize,lemma',
                                 logging_level='WARNING',
                                 download_method=DownloadMethod.REUSE_RESOURCES)

    for index, entry in enumerate(tqdm(input_file)):
        # Get the question and other related data
        nl_question = entry["question"]
        question_identifier = index
        db_id = entry["db_id"]
        query = {"query": entry["query"],
                 "query_toks": entry["query_toks"],
                 "query_toks_no_value": entry["query_toks_no_value"]}

        nl_question_strip = strip_nl(nl_question,
                                     tokenizer=stanza_nlp,
                                     remove_stopwords=False,
                                     lower=False)

        # Process using Stanza
        stanza_processed_nl = stanza_nlp(nl_question)
        logger.debug(nl_question)

        nl_processed = []
        for sentence in stanza_processed_nl.sentences:
            for word in sentence.words:
                nl_processed.append(word.text)

        nl_processed = " ".join(nl_processed)

        # Get sentence embedding
        sentence_embeddings = get_sentence_embedding(nl_processed)

        # Get part-of-speech tagging
        pos_graph_edges = _get_pos_tag_graph(stanza_processed_nl,
                                             sentence_embeddings)
        logger.debug(pos_graph_edges)

        # Get dependency graph
        dependency_graph_edges = _get_dependency_graph(stanza_processed_nl,
                                                       sentence_embeddings)
        logger.debug(dependency_graph_edges)

        spider_nl_question = SpiderQuestionGraph(identifier=question_identifier,
                                                 db_id=db_id,
                                                 query=query,
                                                 question=nl_question,
                                                 question_strip=nl_question_strip,
                                                 vocabulary=vocabulary,
                                                 pos_graph=pos_graph_edges,
                                                 dependency_graph=dependency_graph_edges)

        questions[question_identifier] = spider_nl_question.to_dict()

    return questions
