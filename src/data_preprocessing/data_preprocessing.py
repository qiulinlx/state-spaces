import argparse
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import csv


def binary_dataset_generation(input_csv, output_csv):
    """
    Create a binary dataset from a clustered sequence CSV file.

    Args:
        input_csv (str): Path to the input CSV file containing sequences and clusters.
        output_csv (str): Path to save the output binary dataset CSV file.
    """
    # Load the input CSV file
    df = pd.read_csv(input_csv)
    sequence = df['Sequence']
    Y = df.drop(df.columns[0], axis=1)

    # Define categories (clusters)
    categories = np.arange(0, 500, dtype=float)
    num_rows = df.shape[0]

    # One-hot encode the clusters
    y = []
    for i in range(num_rows):
        row = Y.loc[i].values
        encoded_row = [1 if category in row else 0 for category in categories]
        y.append(encoded_row)

    # Create a DataFrame for the binary encoded data
    binary_df = pd.DataFrame(y)
    max_cluster = binary_df[0]  # Assuming the first column represents the primary cluster

    # Combine sequences and binary labels
    joined_column = pd.concat([sequence, max_cluster], axis=1)
    joined_column.to_csv(output_csv, index=False)


def load_protbert_model(device):
    """
    Load the ProtBERT model and tokenizer.

    Args:
        device (torch.device): Device to load the model on (e.g., 'cuda' or 'cpu').

    Returns:
        model (BertModel): Pretrained ProtBERT model.
        tokenizer (BertTokenizer): Tokenizer for ProtBERT.
    """
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
    model = model.to(device)
    model = model.eval()
    return model, tokenizer


def encode_sequences(sequences, tokenizer, device):
    """
    Encode protein sequences using the ProtBERT tokenizer.

    Args:
        sequences (list): List of protein sequences.
        tokenizer (BertTokenizer): ProtBERT tokenizer.
        device (torch.device): Device to use for encoding.

    Returns:
        input_ids (torch.Tensor): Tokenized input IDs.
        attention_mask (torch.Tensor): Attention masks for the sequences.
    """
    # Add spaces between amino acids and wrap in quotes
    modified_sequences = [' '.join(list(seq)) for seq in sequences]
    modified_sequences = ['"' + seq + '"' for seq in modified_sequences]

    # Tokenize the sequences
    encoded = tokenizer.batch_encode_plus(
        modified_sequences,
        add_special_tokens=True,
        padding=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    return input_ids, attention_mask


def extract_features(model, input_ids, attention_mask):
    """
    Extract features from protein sequences using the ProtBERT model.

    Args:
        model (BertModel): Pretrained ProtBERT model.
        input_ids (torch.Tensor): Tokenized input IDs.
        attention_mask (torch.Tensor): Attention masks for the sequences.

    Returns:
        features (list): List of feature vectors for each sequence.
    """
    with torch.no_grad():
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    features = []
    for seq_num in range(len(embeddings)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_embedding = embeddings[seq_num][1:seq_len - 1]  # Remove [CLS] and [SEP] tokens
        features.append(seq_embedding.cpu().numpy())
    return features


def save_features_to_csv(features, output_csv):
    """
    Save extracted features to a CSV file.

    Args:
        features (list): List of feature vectors.
        output_csv (str): Path to save the CSV file.
    """
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(features)


def main(args):
    """
    Main function to process protein sequences and extract features using ProtBERT.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Create binary dataset
    print(f"Creating binary dataset from {args.input_csv}...")
    binary_dataset_generation(args.input_csv, args.binary_csv)
    print(f"Binary dataset saved to {args.binary_csv}.")

    # Load ProtBERT model and tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading ProtBERT model on {device}...")
    model, tokenizer = load_protbert_model(device)

    # Load binary dataset
    print(f"Loading binary dataset from {args.binary_csv}...")
    df = pd.read_csv(args.binary_csv)
    sequences = df['Sequence'].tolist()

    # Encode sequences
    print("Encoding sequences...")
    input_ids, attention_mask = encode_sequences(sequences, tokenizer, device)

    # Extract features
    print("Extracting features...")
    features = extract_features(model, input_ids, attention_mask)

    # Save features to CSV
    print(f"Saving features to {args.output_csv}...")
    save_features_to_csv(features, args.output_csv)
    print("Feature extraction complete!")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Pipeline to create binary datasets and extract ProtBERT features.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file with clustered sequences.")
    parser.add_argument("--binary_csv", type=str, required=True, help="Path to save the binary dataset CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the ProtBERT features CSV file.")
    args = parser.parse_args()

    # Run the pipeline
    main(args)