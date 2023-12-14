"""
Preprocessing functions to convert Spider questions into graphs
"""

import networkx as nx
from typing import List
import logging
import torch
from torch_geometric.utils.convert import from_networkx

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("SpiderQuestionsToGraph")


class SpiderInputGraphNode:
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


class SpiderInputGraphEdge:
    def __init__(self,
                 source: SpiderInputGraphNode,
                 edge_type: str,
                 target: SpiderInputGraphNode):
        self.source = source
        self.edge_type = edge_type
        self.target = target

    def __str__(self):
        return f'({self.source.name},{self.edge_type},{self.target.name})'

    def __repr__(self):
        return f'({self.source.name},{self.edge_type},{self.target.name})'


class SpiderInputGraph:
    def __init__(self,
                 identifier: str,
                 vocabulary):
        self.identifier = identifier
        self.vocabulary = vocabulary

    def _single_graph_to_networkx(self, graph):
        '''
        Transforms graph into Networkx format and converts into Levi graph
        '''
        networkx_graph = nx.Graph()

        # Connect all nodes in line
        node_names = set()
        node_embeddings = {}
        edge_type_embedding = {}  # Edge type embeddings are initialized randomly
        for edge in graph:
            source_node = edge.source
            target_node = edge.target
            source_node_embedding = source_node.embedding
            target_node_embedding = target_node.embedding
            node_embeddings[source_node.name] = source_node_embedding
            node_embeddings[target_node.name] = target_node_embedding
            node_names.add(source_node)
            node_names.add(target_node)

            if edge.edge_type not in edge_type_embedding:
                edge_type_embedding[edge.edge_type] = torch.randn_like(source_node_embedding)

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
                    print(previous_name)
                    print(current_name)
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
                                        x=edge_type_embedding[edge.edge_type])

            # Add edges
            networkx_graph.add_edge(source_node.name, edge.edge_type)
            networkx_graph.add_edge(target_node.name, edge.edge_type)

        undirected_graph = networkx_graph.to_undirected()
        return undirected_graph

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

    def to_networkx(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def to_pyg(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def to_dict(self):
        raise NotImplementedError("This method should be implemented in a subclass")


class SpiderSchemaGraph(SpiderInputGraph):
    def __init__(self,
                 identifier,
                 graph_edge_list: List[SpiderInputGraphEdge],
                 vocabulary):
        super().__init__(identifier, vocabulary)
        self.graph = graph_edge_list

    def to_networkx(self):
        schema_graph_networkx = self._single_graph_to_networkx(self.graph)
        return schema_graph_networkx

    def to_pyg(self):
        # Obtain the NetworkX representation
        networkx_graph = self.to_networkx()

        # Convert the graph into PyTorch geometric
        pyg_graph = from_networkx(networkx_graph)

        return pyg_graph

    def to_dict(self):
        item_dict = dict()

        item_dict["db_id"] = self.identifier
        item_dict["pyg"] = self.to_pyg()
        #item_dict["networkx"] = self.to_networkx()

        return item_dict

    def __str__(self):
        return f'{self.identifier}: {self.graph}'

    def __repr__(self):
        return f'{self.identifier}: {self.graph}'


class SpiderQuestionGraph(SpiderInputGraph):
    def __init__(self,
                 identifier,
                 db_id: str,
                 query: dict,
                 question: str,
                 question_strip,
                 pos_graph: List[SpiderInputGraphEdge],
                 dependency_graph: List[SpiderInputGraphEdge],
                 vocabulary):
        super().__init__(identifier, vocabulary)
        self.db_id = db_id
        self.query = query
        self.question = question
        self.question_strip = question_strip
        self.pos_graph = pos_graph
        self.dependency_graph = dependency_graph

    def to_networkx(self):
        pos_graph_networkx = self._single_graph_to_networkx(self.pos_graph)
        dependency_graph_networkx = self._single_graph_to_networkx(self.dependency_graph)
        combined_graph_networkx = nx.compose(pos_graph_networkx, dependency_graph_networkx)

        return {"pos": pos_graph_networkx,
                "dependency": dependency_graph_networkx,
                "combined": combined_graph_networkx}

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

    def __str__(self):
        return f'([ID: {self.identifier}] DB: {self.db_id}) {self.question} | {self.pos_graph} | {self.dependency_graph} | Query: {self.query["query"]}'

    def __repr__(self):
        return f'([ID: {self.identifier}] DB: {self.db_id}) {self.question} | {self.pos_graph} | {self.dependency_graph} | Query: {self.query["query"]}'


class SpiderQueryGraph(SpiderInputGraph):
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
        super().__init__(identifier, vocabulary)
        self.db_id = db_id
        self.db_id_original = db_id_original
        self.query = query
        self.root_node = root_node
        self.adjacency_list = adjacency_list
        self.node_type_list = node_type_list
        self.table_tokens = table_tokens
        self.column_tokens = column_tokens

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
        item_dict = {}

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

    def __str__(self):
        return f'([ID: {self.identifier}] DB: {self.db_id}) {self.query} | {self.adjacency_list} | {self.node_type_list}'

    def __repr__(self):
        return f'([ID: {self.identifier}] DB: {self.db_id}) {self.query} | {self.adjacency_list} | {self.node_type_list}'
