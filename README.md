# SQLformer

Official implementation of "SQLformer: Deep Auto-Regressive Query Graph Generation for Text-to-SQL Translation"

# Overview

SQLformer is a novel Transformer architecture specifically crafted to perform text-to-SQL translation tasks. The model predicts SQL queries as abstract syntax trees (ASTs) in an autoregressive way, incorporating structural inductive bias in the encoder and decoder layers. This bias, guided by database table and column selection, aids the decoder in generating SQL query ASTs represented as graphs in a Breadth-First Search canonical order. Comprehensive experiments illustrate the effectiveness of SQLformer on the Spider benchmark.

# Graphical summary of the proposed architecture

![SQLformer](https://github.com/AdrianBZG/SQLformer_Private/assets/8275330/205f393f-d967-47d2-b913-cf95dac2bb04)

# License

SQLformer is released under the [MIT](LICENSE).

# Step-by-step tutorial

In this section, we describe the necessary steps to carry out preprocessing, training and evaluation of SQLformer.

## Environment Setup

To install the required dependencies, follow the below steps.

Install pytorch and pytorch_geometric:

`bash install_pytorch.sh`

Get Poetry shell and install dependencies:

`poetry shell && poetry install`

## Download the data

Please download the official Spider dataset from [here](https://yale-lily.github.io/spider), and unzip it in the data/spider folder.

## Preprocessing

To preprocess the original Spider dataset into SQLformer-compatible training format use:

`make preprocess`

To customize the preprocessing config, please edit the config/configs.py as desired.

## Training

To run training, use:

`make train`

To customize the training config, please edit the config/configs.py as desired.

## Evaluation

To run evaluation, first point to the model checkpoint path by editing the "evaluation" entry in config/configs.py, then run:

`make eval`

# Citation

WIP
