from sklearn import model_selection
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

def load_multiclass_data(data_path):
    """
    Load and preprocess multiclass protein sequence data.

    Args:
        data_path (str): Path to the CSV file containing the dataset.

    Returns:
        X_train (torch.Tensor): Training sequences.
        X_val (torch.Tensor): Validation sequences.
        X_test (torch.Tensor): Testing sequences.
        y_train (torch.Tensor): Training labels.
        y_val (torch.Tensor): Validation labels.
        y_test (torch.Tensor): Testing labels.
        class_weights (torch.Tensor): Class weights for handling imbalanced data.
    """
    df = pd.read_csv(data_path)
    Y = df.drop("Sequence", axis=1)
    categories = np.arange(0, 500, dtype=float)
    num_rows = df.shape[0]

    # Calculate class weights
    counts = [(Y.values == i).sum() for i in categories]
    total_samples = num_rows
    class_weights = [total_samples / (500 * count) if count != 0 else 0.001 for count in counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Encode labels
    y = [[1 if category in row else 0 for category in categories] for row in Y.values]

    # One-hot encode sequences
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
    X = df['Sequence'].apply(lambda seq: np.array([[1 if aa == s else 0 for aa in amino_acids] for s in seq]))
    X = [torch.tensor(arr) for arr in X]
    X = pad_sequence(X, batch_first=True)

    # Split data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=4)
    X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test, y_test, train_size=0.50, test_size=0.50, random_state=4)

    # Convert labels to tensors
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


def load_binary_data(data_path):
    """
    Load and preprocess binary protein sequence data.

    Args:
        data_path (str): Path to the CSV file containing the dataset.

    Returns:
        X_train (torch.Tensor): Training sequences.
        X_val (torch.Tensor): Validation sequences.
        X_test (torch.Tensor): Testing sequences.
        y_train (torch.Tensor): Training labels.
        y_val (torch.Tensor): Validation labels.
        y_test (torch.Tensor): Testing labels.
    """
    df = pd.read_csv(data_path)
    y = df['0']
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

    def one_hot_encode(sequence):
        """
        One-hot encode a protein sequence.

        Args:
            sequence (str): Protein sequence.

        Returns:
            np.ndarray: One-hot encoded sequence.
        """
        encoding = np.zeros((len(sequence), len(amino_acids)))
        for i, aa in enumerate(sequence):
            encoding[i, amino_acids.index(aa)] = 1
        return encoding

    X = df['Sequence'].apply(one_hot_encode)
    seq = [torch.tensor(row) for row in X]
    seq = pad_sequence(seq, batch_first=True)

    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(seq, y, train_size=0.80, test_size=0.20, random_state=4)
    X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test, y_test, train_size=0.50, test_size=0.50, random_state=4)

    # Convert labels to tensors
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test


class ProteinDataset(Dataset):
    """
    Custom PyTorch Dataset for protein sequence data.

    Args:
        x (torch.Tensor): Input sequences.
        y (torch.Tensor): Labels.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.x)

    def __getitem__(self, index):
        """
        Returns a sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (sequence, label) at the given index.
        """
        return self.x[index], self.y[index]