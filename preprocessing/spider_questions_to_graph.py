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
from preprocessing.vocabulary_handler import strip_nl

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("SpiderQuestionsToGraph")


class SpiderNLGraphNode:
    def __init__(self,
                 name: str,
                 type: str,
                 embedding=None):
        self.name = name
        self.type = type
        self.embedding = embedding

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return f'{self.name}'


class SpiderNLGraphEdge:
    def __init__(self,
                 source: SpiderNLGraphNode,
                 edge_type: str,
                 target: SpiderNLGraphNode):
        self.source = source
        self.edge_type = edge_type
        self.target = target

    def __str__(self):
        return f'({self.source.name},{self.edge_type},{self.target.name})'

    def __repr__(self):
        return f'({self.source.name},{self.edge_type},{self.target.name})'


class SpiderNLQuestion:
    def __init__(self,
                 identifier: int,
                 db_id: str,
                 query: dict(),
                 question: str,
                 question_strip,
                 pos_graph: List[SpiderNLGraphEdge],
                 dependency_graph: List[SpiderNLGraphEdge],
                 vocabulary):
        self.identifier = identifier
        self.db_id = db_id
        self.query = query
        self.question = question
        self.question_strip = question_strip
        self.pos_graph = pos_graph
        self.dependency_graph = dependency_graph
        self.vocabulary = vocabulary

    def __str__(self):
        return f'([ID: {self.identifier}] DB: {self.db_id}) {self.question} | {self.pos_graph} | {self.dependency_graph} | Query: {self.query["query"]}'

    def __repr__(self):
        return f'([ID: {self.identifier}] DB: {self.db_id}) {self.question} | {self.pos_graph} | {self.dependency_graph} | Query: {self.query["query"]}'

    def _single_graph_to_networkx(self, graph):
        '''
        Transforms graph into Networkx format and converts into Levi graph
        '''
        networkx_graph = nx.Graph()

        # Connect all nodes in line
        node_names = set()
        node_embeddings = {}
        edge_type_embeddings = {}  # Edge type node embeddings are initialized randomly
        for edge in graph:
            source_node = edge.source
            target_node = edge.target
            node_embeddings[source_node.name] = source_node.embedding
            node_embeddings[target_node.name] = target_node.embedding
            node_names.add(source_node)
            node_names.add(target_node)
            if edge.edge_type not in edge_type_embeddings:
                edge_type_embeddings[edge.edge_type] = torch.randn_like(source_node.embedding)

        for previous, current in zip(list(node_names), list(node_names)[1:]):
            previous_name = previous.name
            current_name = current.name
            if previous_name not in networkx_graph:
                if node_embeddings[previous_name] is None:
                    print(previous_name)
                    raise ValueError('test1')
                networkx_graph.add_node(previous_name,
                                        x=node_embeddings[previous_name])

            if current_name not in networkx_graph:
                if node_embeddings[current_name] is None:
                    raise ValueError('test2')
                networkx_graph.add_node(current_name,
                                        x=node_embeddings[current_name])

            networkx_graph.add_edge(previous_name, current_name)

        # Loop over relations
        for edge in graph:
            source_node = edge.source
            target_node = edge.target

            # Add nodes if not exist
            if source_node.name not in networkx_graph:
                if node_embeddings[source_node.name] is None:
                    raise ValueError('test3')
                networkx_graph.add_node(source_node.name,
                                        x=node_embeddings[source_node.name])

            if target_node.name not in networkx_graph:
                if node_embeddings[target_node.name] is None:
                    raise ValueError('test4')
                networkx_graph.add_node(target_node.name,
                                        x=node_embeddings[target_node.name])

            if edge.edge_type not in networkx_graph:
                networkx_graph.add_node(edge.edge_type,
                                        x=edge_type_embeddings[edge.edge_type])

            # Add edges
            networkx_graph.add_edge(source_node.name, edge.edge_type)
            networkx_graph.add_edge(target_node.name, edge.edge_type)

        undirected_graph = networkx_graph.to_undirected()
        return undirected_graph

    def to_networkx(self):
        pos_graph_networkx = self._single_graph_to_networkx(self.pos_graph)
        dependency_graph_networkx = self._single_graph_to_networkx(self.dependency_graph)
        combined_graph_networkx = nx.compose(pos_graph_networkx, dependency_graph_networkx)

        return {"pos": pos_graph_networkx,
                "dependency": dependency_graph_networkx,
                "combined": combined_graph_networkx}

    def _single_graph_to_pyg(self, graph):
        # Obtain the NetworkX representation
        networkx_graph = self._single_graph_to_networkx(graph)

        # Convert the graph into PyTorch geometric
        pyg_graph = from_networkx(networkx_graph)

        return pyg_graph

    def _combined_graph_to_pyg(self, graph1, graph2):
        # Get combined NetworkX object
        graph1_networkx = self._single_graph_to_networkx(graph1)
        graph2_networkx = self._single_graph_to_networkx(graph2)
        combined_graph_networkx = nx.compose(graph1_networkx, graph2_networkx)

        # Convert the graph into PyTorch geometric
        combined_graph_pyg = from_networkx(combined_graph_networkx)

        return combined_graph_pyg

    def to_pyg(self):
        pos_graph_pyg = self._single_graph_to_pyg(self.pos_graph)
        dependency_graph_pyg = self._single_graph_to_pyg(self.dependency_graph)
        combined_graph_pyg = self._combined_graph_to_pyg(self.pos_graph,
                                                         self.dependency_graph)

        return {"pos": pos_graph_pyg,
                "dependency": dependency_graph_pyg,
                "combined": combined_graph_pyg}

    def to_dict(self):
        item_dict = dict()

        item_dict["identifier"] = self.identifier
        item_dict["db_id"] = self.db_id
        item_dict["query"] = self.query
        item_dict["question"] = self.question
        item_dict["question_strip"] = self.question_strip
        item_dict["vocabulary"] = self.vocabulary
        item_dict["pyg"] = self.to_pyg()
        #item_dict["networkx"] = self.to_networkx()

        return item_dict


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
        word_node = SpiderNLGraphNode(word, "word", embedding=sentence_embeddings[word])
        pos_tag_node = SpiderNLGraphNode(pos, "pos_tag", embedding=torch.randn_like(sentence_embeddings[word]))
        word_pos_edge = SpiderNLGraphEdge(word_node, "has_pos", pos_tag_node)
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

        source_word_node = SpiderNLGraphNode(dependency_source.text, "word",
                                             embedding=sentence_embeddings[dependency_source.text])
        target_word_node = SpiderNLGraphNode(dependency_target.text, "word",
                                             embedding=sentence_embeddings[dependency_target.text])
        dependency_edge = SpiderNLGraphEdge(source_word_node, dependency_type, target_word_node)
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

        print(nl_question_strip)
        print(nl_question)

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

        spider_nl_question = SpiderNLQuestion(identifier=question_identifier,
                                              db_id=db_id,
                                              query=query,
                                              question=nl_question,
                                              question_strip=nl_question_strip,
                                              vocabulary=vocabulary,
                                              pos_graph=pos_graph_edges,
                                              dependency_graph=dependency_graph_edges)

        questions[question_identifier] = spider_nl_question.to_dict()

    return questions
