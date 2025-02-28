# Protein Annotation using Structured State Space Models (S4)

## Overview

Approximately 100,000 unique proteins exist in the human body alone, with many more in other organisms. These proteins perform vital functions necessary for life, making it crucial to understand their roles. While many proteins have been sequenced, only a fraction have been annotated with their functions due to the labor-intensive nature of the process. This leaves a significant number of sequenced proteins with unknown functions that could be invaluable to researchers.

Structured State Space Models (S4), introduced by [Gu et al. (2022)](https://arxiv.org/abs/2111.00396), have demonstrated superior performance in handling long sequential data compared to other machine learning models. The S4 model outperforms Transformers and other sequence modeling architectures on the Long Range Arena benchmark, making it particularly well-suited for protein sequences, which can span up to 35,000 amino acids. This repository contains the code and implementation details from the accompanying [paper](results/Honours_Thesis.pdf).

---

## Data Preprocessing

To preprocess the protein sequence data and extract features using ProtBERT, run the following command:

```bash
python protbert_pipeline.py \
    --input_csv ClusteredSeq1.csv \
    --binary_csv Binaryset.csv \
    --output_csv ProtBertdata.csv
```
This script performs the following:
    <!-- Converts clustered protein sequences into a binary dataset.

    Extracts feature embeddings using the ProtBERT model.

    Saves the extracted features to a CSV file. -->

## How to Run
### Binary Classification

To train the S4 model for binary classification, use the following command:

```bash
python train_s4_binary.py \
    --batch_size 2 \
    --epochs 10 \
    --steps 50 \
    --lr 0.001 \
    --weight_decay 0.0000001 \
    --d_model 512 \
    --n_layers 10 \
    --data_path Binaryset.csv \
    --wandb_key YOUR_WANDB_KEY
```

### Multilabel Classification

To train the S4 model for multilabel classification, use the following command:

```bash
python train_s4_multilabel.py \
    --batch_size 12 \
    --epochs 10 \
    --steps 50 \
    --lr 0.0015 \
    --weight_decay 0.000001 \
    --d_model 512 \
    --n_layers 25 \
    --data_path subsetdata.csv \
    --wandb_key YOUR_WANDB_KEY
```


