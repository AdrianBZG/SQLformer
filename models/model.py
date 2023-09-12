import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import torch.nn.functional as F
from models.encoders.gnn_encoder_factory import get_gnn_encoder
from models.utils import xavier_init_weights
from models.layers import PositionalEncoding
from models.decoders.transformer_decoder import TransformerDecoder
from models.encoders.transformer_encoder import TransformerEncoder
from data.dataset import AST_NODE_TYPES
from utils import pad_and_mask_combined_graph, construct_future_mask
from torch import multiprocessing as torchmultiprocessing
torchmultiprocessing.set_sharing_strategy('file_system')

PRECOMPUTED_NODE_EMBEDDING_SIZE = 1024
NUMBER_QUERY_NODE_TYPES = len(AST_NODE_TYPES)


class Model(nn.Module):
    def __init__(self,
                 device,
                 vocabulary,
                 tables_vocab,
                 columns_vocab,
                 config):
        super(Model, self).__init__()

        self.max_prev_node = config.get('max_prev_bfs_node')
        self.device = device
        self.vocabulary = vocabulary
        self.tables_vocab = tables_vocab
        self.columns_vocab = columns_vocab
        self.max_query_ast_nodes = config.get('max_query_ast_nodes')
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 1.0)
        self.k_tables = config.get('k_tables')
        self.k_columns = config.get('k_columns')

        self.schema_encoder = get_gnn_encoder(config.get('encoder_schema_gnn_type'),
                                              in_channels=PRECOMPUTED_NODE_EMBEDDING_SIZE,
                                              out_channels=config.get('encoder_schema_hidden_dim'),
                                              params={"heads": config.get('encoder_schema_num_heads'),
                                                      "layers": config.get('encoder_schema_num_layers'),
                                                      "dropout": config.get('dropout')})

        self.question_graph_encoder = get_gnn_encoder(config.get('encoder_question_gnn_type'),
                                                      in_channels=PRECOMPUTED_NODE_EMBEDDING_SIZE,
                                                      out_channels=config.get('encoder_question_hidden_dim'),
                                                      params={"heads": config.get('encoder_question_num_heads'),
                                                              "layers": config.get('encoder_question_num_layers'),
                                                              "dropout": config.get('dropout')})

        self.output_vocab_dim = NUMBER_QUERY_NODE_TYPES \
                                + len(tables_vocab.as_list()) \
                                + len(columns_vocab.as_list()) \
                                + len(["@val"])

        # Projection/embedding for adjacency vectors and node types
        self.node_adj_linear = Linear(config.get('max_prev_bfs_node'),
                                      config.get('decoder_node_adj_emb_hidden_dim'))

        self.node_type_linear = Linear(self.output_vocab_dim,
                                       config.get('decoder_node_types_emb_hidden_dim'))

        # Positional embeddings
        self.src_positional_encoding = PositionalEncoding(d_model=config.get('encoder_question_hidden_dim'),
                                                          max_len=config.get('max_question_graph_n_nodes'))

        self.trg_positional_encoding = PositionalEncoding(d_model=config.get('decoder_node_adj_emb_hidden_dim') +
                                                                  config.get('decoder_node_types_emb_hidden_dim'),
                                                          max_len=config.get('max_query_ast_nodes'))

        # Transformer encoder
        self.transformer_enc = TransformerEncoder(hidden_dim=config.get('encoder_question_hidden_dim'),
                                                  ff_dim=config.get('decoder_hidden_dim'),
                                                  num_heads=config.get('num_heads'),
                                                  num_layers=config.get('decoder_num_layers'),
                                                  dropout=config.get('dropout'))

        # Transformer decoder
        self.transformer_dec = TransformerDecoder(hidden_dim=config.get('decoder_node_adj_emb_hidden_dim'),
                                                  num_layers=config.get('decoder_num_layers'),
                                                  num_heads=config.get('num_heads'),
                                                  ff_dim=config.get('decoder_hidden_dim'),
                                                  dropout_p=config.get('dropout'))

        # Learnable table token
        self.table_token = nn.Parameter(torch.rand(1, config.get('encoder_question_hidden_dim')))

        # Learnable column token
        self.column_token = nn.Parameter(torch.rand(1, config.get('encoder_question_hidden_dim')))

        # Selection heads for tables and columns
        self.mlp_tables = Linear(config.get('encoder_question_hidden_dim'),
                                 len(tables_vocab.as_list()))

        self.mlp_columns = Linear(config.get('encoder_question_hidden_dim'),
                                  len(columns_vocab.as_list()))

        # Table Key-Embedding for tables and columns
        self.tables_embedding = nn.Embedding(len(tables_vocab.as_list()),
                                             config.get('encoder_question_hidden_dim'))

        self.columns_embedding = nn.Embedding(len(columns_vocab.as_list()),
                                              config.get('encoder_question_hidden_dim'))

        # Decoder output heads
        self.adj_out_1 = Linear(config.get('decoder_node_adj_emb_hidden_dim'),
                                config.get('decoder_node_adj_emb_hidden_dim'))
        self.adj_out_2 = Linear(config.get('decoder_node_types_emb_hidden_dim'),
                                config.get('max_prev_bfs_node'))

        self.types_out_1 = Linear(config.get('decoder_node_types_emb_hidden_dim'),
                                  config.get('decoder_node_types_emb_hidden_dim'))
        self.types_out_2 = Linear(config.get('decoder_node_types_emb_hidden_dim'),
                                  self.output_vocab_dim)

        self.apply(xavier_init_weights)

    def forward(self, batch):
        device = self.device
        to_device = lambda x: Batch.from_data_list([y.to(device) for y in batch[x]])

        x_adj, x_node_types = batch["x_adj"].to(device), batch["x_node_types"].to(device)
        combined_question_graph, schema = to_device("questions_combined_graph"), to_device("schemas")

        combined_question_graph_z, _ = self.question_graph_encoder(combined_question_graph.x,
                                                                   combined_question_graph.edge_index,
                                                                   combined_question_graph.batch)

        _, schema_g = self.schema_encoder(schema.x,
                                          schema.edge_index,
                                          schema.batch)

        pad_graph_batch, pad_mask, pad_seq_len = pad_and_mask_combined_graph(unbatch(combined_question_graph_z,
                                                                                     combined_question_graph.batch))

        schema_g_rep = schema_g.unsqueeze(1).repeat(1, pad_graph_batch.shape[1], 1)
        combined_graph_w_schema = pad_graph_batch + schema_g_rep

        add_tokens = lambda x, token: torch.stack([torch.vstack((token, x[i])) for i in range(len(x))])
        combined_graph_w_tokens = add_tokens(combined_graph_w_schema, self.table_token + self.column_token)

        question_encoded = self.transformer_enc(self.src_positional_encoding(combined_graph_w_tokens), pad_mask)
        X_tables, X_columns, X_question = question_encoded[:, 0], question_encoded[:, 1], question_encoded[:, 2:]

        get_top_k = lambda logits, k, embedding: embedding(F.softmax(logits, dim=-1).topk(k, dim=1)[1]).mean(1)
        Xk_tables, Xk_columns = get_top_k(self.mlp_tables(X_tables), self.k_tables, self.tables_embedding), get_top_k(
            self.mlp_columns(X_columns), self.k_columns, self.columns_embedding)

        X_schema_hat, X_question_hat = torch.cat([Xk_tables, Xk_columns], dim=1), torch.einsum('bij,blm->blm',
                                                                                               torch.cat([Xk_tables,
                                                                                                          Xk_columns],
                                                                                                         dim=1),
                                                                                               X_question)

        apply_linear = lambda x, layer: layer(F.relu(x))
        x_adj_embed, x_node_types_embed = apply_linear(x_adj, self.node_adj_linear), apply_linear(x_node_types,
                                                                                                  self.node_type_linear)

        future_mask_adj = construct_future_mask(x_adj_embed.shape[1], device=device)
        decoded_adj, decoded_nt = self.transformer_dec(X_question_hat, pad_mask[:, 2:], x_adj_embed, future_mask_adj,
                                                       x_node_types_embed)

        output_adj, output_node_type = apply_linear(decoded_adj, self.adj_out_2), apply_linear(decoded_nt,
                                                                                               self.types_out_2)

        return output_adj, output_node_type, self.mlp_tables(X_tables), self.mlp_columns(X_columns), pad_seq_len

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reset_hidden(self):
        pass
