configs = dict()
configs["preprocessing"] = {}
configs["training"] = {}


configs["preprocessing"]["base"] = {"generate_schema": False,
                                    "generate_questions": True,
                                    "generate_queries": False,
                                    "init_model": 'roberta-base',  # ['all-roberta-large-v1', 'grappa_large_jnt']
                                    "root_path": "data/spider",
                                    "splits": ["train_spider"],
                                    "output_path": "data/spider"}

configs["training"]["base"] = {"root_path": "data/spider",
                               "max_prev_bfs_node": 30,
                               "max_query_ast_nodes": 200,
                               "max_question_graph_n_nodes": 64,
                               "run_training": True,
                               "run_validation": True,
                               "batch_size": 16,
                               "num_dataloader_workers": 0,
                               "num_epochs": 100000,
                               "encoder_schema_gnn_type": "gat",
                               "encoder_schema_hidden_dim": 512,
                               "encoder_schema_num_heads": 8,
                               "encoder_schema_num_layers": 6,
                               "encoder_question_gnn_type": "gat",
                               "encoder_question_hidden_dim": 512,
                               "encoder_question_num_heads": 8,
                               "encoder_question_num_layers": 6,
                               "cross_attention_num_heads": 4,
                               "bottleneck_encoder_hidden_dim": 256,
                               "bottleneck_encoder_num_layers": 2,
                               "bottleneck_encoder_concepts_embedding_dim": 256,
                               "decoder_hidden_dim": 1024,
                               "decoder_num_layers": 6,
                               "decoder_node_types_emb_hidden_dim": 512,
                               "decoder_node_adj_emb_hidden_dim": 512,
                               "num_heads": 8,
                               "dropout": 0.1,
                               "k_tables": 20,
                               "k_columns": 20,
                               "optimizer": "adam",
                               "learning_rate": 1e-3,
                               "weight_decay": 5e-6,
                               "teacher_forcing_ratio": 1.0,
                               "loss_lambda": [1, 1, 1, 1],
                               "model_output_path": "model_checkpoints"}

configs["evaluation"] = {"root_path": "data/spider",
                         "model_path": "model_checkpoints/SQLformer_step_20000"}
