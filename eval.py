"""
Module that contains functions for calculating performances during SQLformer training
"""

import torch
from tqdm import tqdm
from models.utils import sample_sigmoid, adjust_softmax
from config.configs import configs


def calculate_accuracy_helper(batch_labels, batch_outputs, seq_len, is_adj=False):
    batch_accuracies = []
    for binx, (labels, outputs) in enumerate(zip(batch_labels, batch_outputs)):
        accuracy = 1.0
        for inx in range(labels.shape[1]):
            if inx >= seq_len[binx]:
                break

            if inx >= len(outputs) or inx >= len(labels):
                if is_adj:
                    accuracy = 0.0
                else:
                    continue

            prediction = outputs[inx] if is_adj else torch.argmax(outputs[inx])
            target = labels[inx] if is_adj else torch.argmax(labels[inx])
            if not torch.equal(prediction, target):
                accuracy = 0.0
                if is_adj:
                    break

        batch_accuracies.append(accuracy)
    return sum(batch_accuracies) / len(batch_accuracies)


def calculate_batch_accuracies(output_adj, y_adj, output_node_type, y_node_types, seq_len):
    # Adjust outputs for accuracy calculation
    output_adj = sample_sigmoid(torch.sigmoid(output_adj), output_adj.device, sample=False, threshold=0.5)
    output_node_type = adjust_softmax(output_node_type, output_node_type.device)

    batch_accuracy_adj = calculate_accuracy_helper(y_adj, output_adj, seq_len, is_adj=True)
    batch_accuracy_types = calculate_accuracy_helper(y_node_types, output_node_type, seq_len, is_adj=False)

    batch_accuracies_all = []
    for binx, (adj_label, adj_output, types_label, types_output) in enumerate(zip(y_adj, output_adj,
                                                                                  y_node_types, output_node_type)):
        adj_accuracy = calculate_accuracy_helper([adj_label], [adj_output], [seq_len[binx]], is_adj=True)
        types_accuracy = calculate_accuracy_helper([types_label], [types_output], [seq_len[binx]], is_adj=False)
        batch_accuracies_all.append(adj_accuracy if adj_accuracy == types_accuracy else 0.0)

    batch_accuracy_all = sum(batch_accuracies_all) / len(batch_accuracies_all)

    return batch_accuracy_adj, batch_accuracy_types, batch_accuracy_all


def run_inference(dataloader, model, verbose=False):
    """
    Method to run an inference using the given models on the specified data loader. This generates the graph
    timestep by timestep, as it should be during inference time.
    """

    model.eval()

    batch_accuracies_adj = []
    batch_accuracies_types = []
    batch_accuracies_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            output_adj, y_adj, output_node_type, y_node_types, output_seq_len = model.evaluation(batch, verbose)

            # Limit graph to the sequence length
            output_adj = output_adj[:, :output_seq_len - 1, :]
            output_node_type = output_node_type[:, :output_seq_len - 1, :]

            # Calculate accuracy for this batch
            batch_accuracy_adj, batch_accuracy_types, batch_accuracy_all = calculate_batch_accuracies(output_adj,
                                                                                                      y_adj,
                                                                                                      output_node_type,
                                                                                                      y_node_types)

            batch_accuracies_adj.append(batch_accuracy_adj)
            batch_accuracies_types.append(batch_accuracy_types)
            batch_accuracies_all.append(batch_accuracy_all)

    batch_accuracy_adj = sum(batch_accuracies_adj) / len(batch_accuracies_adj)
    batch_accuracy_types = sum(batch_accuracies_types) / len(batch_accuracies_types)
    batch_accuracy_all = sum(batch_accuracies_all) / len(batch_accuracies_all)
    return batch_accuracy_all, batch_accuracy_adj, batch_accuracy_types


if __name__ == "__main__":
    evaluation_config = configs["evaluation"]
    run_inference(evaluation_config)
