"""
Main training pipeline module
"""

import os
import torch
from torch import multiprocessing as torchmultiprocessing
torchmultiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm, trange
import argparse
from transformers import get_scheduler

from config.configs import configs
from data.dataset import get_dataset, get_data_loader
from models.model import Model
from utils import create_optimizer, calculate_loss, set_seed
from eval import calculate_batch_accuracies

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("train")


def run_epoch(config, dataloader, model, optimizer, scheduler, evaluation=False, epoch=None):
    model.eval() if evaluation else model.train()

    metrics = {
        "losses": {"all": [], "adj": [], "node_types": [], "table_concepts": [], "column_concepts": []},
        "batch_accuracies": {"adj": [], "types": [], "all": []}
    }

    data_iterator = tqdm(dataloader)
    for batch_idx, batch in enumerate(data_iterator):
        to_device = lambda x: batch[x].to(model.device)
        y_adj, y_node_types, y_table_targets, y_column_targets, x_adj_padding_mask = map(to_device, ["y_adj", "y_node_types", "y_table_targets", "y_column_targets", "x_adj_padding_mask"])

        output_adj, output_node_type, tables_logits, columns_logits, seq_len = model(batch)

        if not evaluation:
            losses = calculate_loss(x_adj_padding_mask, y_adj, output_adj, output_node_type, y_node_types, tables_logits, y_table_targets, columns_logits, y_column_targets, config)
            for key, loss in zip(metrics["losses"].keys(), losses):
                metrics["losses"][key].append(loss.item())

            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        accuracies = calculate_batch_accuracies(output_adj, y_adj, output_node_type, y_node_types, seq_len)
        for key, acc in zip(metrics["batch_accuracies"].keys(), accuracies):
            metrics["batch_accuracies"][key].append(acc)

        data_iterator.set_postfix({'loss': "{:.8f}".format(metrics["losses"]["all"][-1])})

    return metrics


def log_epoch_summary(epoch, metrics):
    if epoch is None:
        return
    avg = lambda x: sum(x) / len(x)
    logger.info("======================================================")
    logger.info(f'ACCURACY STATS (EPOCH: {epoch})')
    for key in metrics["batch_accuracies"].keys():
        logger.info(f'Batch Accuracy ({key.capitalize()}): {avg(metrics["batch_accuracies"][key])}')
    logger.info("======================================================")


def prepare_model_and_optimizer(vocabulary, tables_vocab, columns_vocab, config, device):
    model = Model(vocabulary=vocabulary,
                  tables_vocab=tables_vocab,
                  columns_vocab=columns_vocab,
                  config=config,
                  device=device
                  ).to(device)

    model_parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"The model has {model_parameters_count} trainable parameters")

    optimizer = create_optimizer(config.get('optimizer'), model, config)

    return model, optimizer


def run_training(config):
    logger.info(f"Running training with config: {config}")
    root_path = config.get('root_path')

    # Train data
    train_dataset, vocabulary, tables_vocab, columns_vocab = get_dataset(root_path,
                                                                         split="train_spider",
                                                                         max_prev_node=config.get('max_prev_bfs_node'))

    #if config.get('run_validation', False):
    #    # Validation data
    #    val_dataset, val_vocabulary, val_tables_vocab, val_columns_vocab = get_dataset(root_path,
    #                                                                                   split="dev",
    #                                                                                   max_prev_node=config.get(
    #                                                                                       'max_prev_bfs_node'))

    # Create the model and optimizer
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    device = "cpu"

    model, optimizer = prepare_model_and_optimizer(vocabulary,
                                                   tables_vocab,
                                                   columns_vocab,
                                                   config,
                                                   device=device)

    if config.get('run_training', True):
        train_dataloader = get_data_loader(train_dataset,
                                           batch_size=config.get('batch_size'),
                                           shuffle=True,
                                           num_workers=config.get('num_dataloader_workers'),
                                           tables_vocab=tables_vocab,
                                           columns_vocab=columns_vocab)

        num_training_steps = len(train_dataloader) // config.get('gradient_accumulation_steps') * config.get('num_epochs')
        num_warmup_steps = int(num_training_steps * config.get('warmup_proportion'))

        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Report training info
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_dataset))
        logging.info("  Num Epochs = %d", config.get('num_epochs'))
        logging.info("  Warmup steps = %d", num_warmup_steps)
        logging.info("  Total optimization steps = %d", num_training_steps)

        # Run training for the desired number of epochs
        optimizer.zero_grad()
        set_seed(config.get('seed'))

        model.train()
        epoch_iterator = trange(int(config.get('num_epochs')))
        for epoch in epoch_iterator:
            # Train loop
            epoch_iterator.set_postfix({'epoch': epoch + 1})

            metrics = run_epoch(config,
                                train_dataloader,
                                model,
                                optimizer,
                                scheduler,
                                evaluation=False,
                                epoch=epoch)

            log_epoch_summary(epoch, metrics)

            if config.get('run_validation', False):
                with torch.no_grad():
                    # Validation loop
                    logger.debug(f"RUNNING VALIDATION FOR EPOCH #{epoch + 1}")
                    val_dataloader = get_data_loader(train_dataset,
                                                     batch_size=config.get('batch_size'),
                                                     shuffle=False,
                                                     num_workers=config.get('num_dataloader_workers'),
                                                     tables_vocab=tables_vocab,
                                                     columns_vocab=columns_vocab)

                    metrics = run_epoch(config,
                                        val_dataloader,
                                        model,
                                        optimizer,
                                        scheduler,
                                        evaluation=True,
                                        epoch=epoch)

                    log_epoch_summary(epoch, metrics)

                    # Save model to disk
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optim': optimizer.state_dict()
                    }, open(os.path.join(config.get('model_output_path'), f'SQLformer_step_{epoch}.bin'), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--config_name', type=str, nargs='?', help='The training config to use', required=True)
    args = parser.parse_args()

    if args.config_name not in configs["training"]:
        raise ValueError(f"Training config {args.config_name} does not exist")

    run_training(configs["training"][args.config_name])
