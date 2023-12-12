import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.nn.utils.rnn import pad_sequence


bts=12 #Batch size
n_classes=500
d_input=21 #Length of each vector in sequence
d_output = 1
steps =2
n_epochs=2
print('Imported packages successfully')

#df=pd.read_csv('ClusteredSeq.csv')

df=pd.read_csv('subsetdata.csv')

df=df.drop(['Unnamed: 0'], axis=1)
Y = df.drop("Sequence", axis=1)

# Initialize a dictionary to store the counts
counts=[]
for i in range(0,500):
    label = float(i)
    count= (Y.values == label).sum()
    print(count)
    counts.append(count)


# Step 2: Calculate class weights based on class frequencies
total_samples = 998
class_weights=[]
for i in range(len(counts)):
    if counts[i] == 0:
        weights = 0.001
    else:
    #print(class_labels[i])
        weights = total_samples / (n_classes* counts[i])
    
    class_weights.append(weights)





# # Categories
# categories =np.array(np.arange(0,500), dtype= float) #Get a list of all the clusters
# num_rows = df.shape[0] #no. samples in dataset

# #One-hot encoding
# y = []

# for i in range(num_rows):
#     '''Encoding each protein sequence's GO annotation as a binary vector of length equal to the number of clusters'''
#     row=Y.loc[i]
#     row = row.values

#     encoded_row = [1 if category in row else 0 for category in categories]
#     y.append(encoded_row)

# amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'Y']

# def one_hot_encode(sequence):
#     '''Produce one hot encoding of the protein sequences'''
#     encoding = np.zeros((len(sequence), len(amino_acids)))
#     for i, aa in enumerate(sequence):
#         encoding[i, amino_acids.index(aa)] = 1
#     return encoding

# X= df['Sequence'].apply(one_hot_encode)
# seq=[]
# for row in X:
#     seq.append(row)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(seq, y,
#                                     train_size=0.80, test_size=0.20, random_state=4)

# X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test, y_test, train_size=0.50, test_size=0.50, random_state=4)

# print(y_train)

# from torch.utils.data import Dataset

# X_train = [torch.tensor(arr) for arr in X_train]
# X_train = pad_sequence(X_train, batch_first=True)


# X_test=[torch.tensor(arr) for arr in X_test]
# X_test=pad_sequence(X_test, batch_first=True)

# X_val=[torch.tensor(arr) for arr in X_val]
# X_val=pad_sequence(X_val, batch_first=True)


# y_train=torch.tensor(y_train,  dtype=torch.float32)
# y_test=torch.tensor(y_test,  dtype=torch.float32)
# y_val=torch.tensor(y_val,  dtype=torch.float32)

# class CustomDataset(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, index):
#         return self.x[index], self.y[index]

# # Create an instance of your custom dataset
# trainset = CustomDataset(X_train, y_train)
# valset = CustomDataset(X_val, y_val)
# testset = CustomDataset(X_test, y_test)

# # Dataloaders
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=bts, shuffle=True, num_workers=5)
# valloader = torch.utils.data.DataLoader(valset, batch_size=bts, shuffle=False, num_workers=5)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=5)

