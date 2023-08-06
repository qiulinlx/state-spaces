import pandas as pd
import numpy as np
import torch

df=pd.read_csv('Clusteredseq.csv')
Proteindata=pd.DataFrame()
X=df[['1']]
Proteindata['AAseq'] = X
# Function to split a string into individual letters
def split_letters(s):
    return [char for char in s]

# Tokenize: Apply the split_letters function to each element in the DataFrame
Sequences = X.applymap(split_letters)

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'Y']

# One-hot encode each sequence
def one_hot_encode(sequence):
    encoding = np.zeros((len(sequence), len(amino_acids)))
    for i, aa in enumerate(sequence):
        encoding[i, amino_acids.index(aa)] = 1
    return encoding

from torch.nn.utils.rnn import pad_sequence
test= df['1'].apply(one_hot_encode)
# Convert the one-hot encoded sequences to PyTorch tensors
X_train = test.apply(lambda x: torch.tensor(x, dtype=torch.float32))
X_train_padded = pad_sequence(X_train, batch_first=True)

