configs = dict()
configs["preprocessing"] = {}
configs["training"] = {}


configs["preprocessing"]["base"] = {"generate_schema": False,
                                    "generate_questions": False,
                                    "generate_queries": True,
                                    "init_model": 'roberta-base',  # ['all-roberta-large-v1', 'grappa_large_jnt']
                                    "root_path": "data/spider",
                                    "splits": ["train_spider"],
                                    "output_path": "data/spider"}

configs["training"]["base"] = {"root_path": "data/spider",
                               "max_prev_bfs_node": 30,
                               "max_query_ast_nodes": 200,
                               "max_question_graph_n_nodes": 64,
                               "run_training": True,
                               "run_validation": False,
                               "batch_size": 16,
                               "gradient_accumulation_steps": 1,
                               "fp16": False,
                               "seed": 39342211,
                               "num_dataloader_workers": 0,
                               "num_epochs": 20,
                               "encoder_input_dim": 768,
                               "encoder_schema_gnn_type": "gat",
                               "encoder_schema_hidden_dim": 512,
                               "encoder_schema_num_heads": 8,
                               "encoder_schema_num_layers": 4,
                               "encoder_question_gnn_type": "gat",
                               "encoder_question_hidden_dim": 512,
                               "encoder_question_num_heads": 8,
                               "encoder_question_num_layers": 4,
                               "cross_attention_num_heads": 4,
                               "bottleneck_encoder_hidden_dim": 256,
                               "bottleneck_encoder_num_layers": 2,
                               "bottleneck_encoder_concepts_embedding_dim": 256,
                               "decoder_hidden_dim": 512,
                               "decoder_num_layers": 4,
                               "decoder_node_types_emb_hidden_dim": 512,
                               "decoder_node_adj_emb_hidden_dim": 512,
                               "num_heads": 8,
                               "dropout": 0.1,
                               "k_tables": 5,
                               "k_columns": 10,
                               "optimizer": "adamw",
                               "learning_rate": 3e-5,
                               "eps": 1e-8,
                               "warmup_proportion": 0.1,
                               "teacher_forcing_ratio": 1.0,
                               "loss_lambda": [1, 1, 1, 1],
                               "model_output_path": "model_checkpoints"}

configs["evaluation"] = {"root_path": "data/spider",
                         "model_path": "model_checkpoints/SQLformer_step_20000"}
