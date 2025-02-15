import pandas as pd
import numpy as np
import os

os.environ['CXX'] = 'cl.exe'

from sklearn import model_selection
import torch.nn as nn
from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
import torch.optim as optim
import torch
from torch.nn.utils.rnn import pad_sequence
from torcheval.metrics import MulticlassF1Score, MultilabelAUPRC
import wandb

bts=12 #Batch size
n_classes=500
d_input=21 #Length of each vector in sequence
d_output = 1
steps =50
epochs=10

wandb.login(key='7b95dbe82c6138a403e12795e0fd55461555b0e4',)

print('Imported packages successfully')

df=pd.read_csv('subsetdata.csv')

#df=pd.read_csv('subsetdata.csv')

#df=df.drop(['Unnamed: 0'], axis=1)
Y = df.drop("Sequence", axis=1)

# Categories
categories =np.array(np.arange(0,500), dtype= float) #Get a list of all the clusters
num_rows = df.shape[0] #no. samples in dataset

# Initialize a dictionary to store the counts
counts=[]
for i in range(0,n_classes):
    label = float(i)
    count= (Y.values == label).sum()
    counts.append(count)

# Step 2: Calculate class weights based on class frequencies
total_samples = num_rows
class_weights=[]
for i in range(len(counts)):
    if counts[i] == 0:
        weights = 0.001
    else:
    #print(class_labels[i])
        weights = total_samples / (n_classes* counts[i])
    
    class_weights.append(weights)

class_weights=torch.tensor(class_weights, dtype=torch.float32)
#One-hot encoding
y = []

for i in range(num_rows):
    '''Encoding each protein sequence's GO annotation as a binary vector of length equal to the number of clusters'''
    row=Y.loc[i]
    row = row.values

    encoded_row = [1 if category in row else 0 for category in categories]
    y.append(encoded_row)

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'Y']

def one_hot_encode(sequence):
    '''Produce one hot encoding of the protein sequences'''
    encoding = np.zeros((len(sequence), len(amino_acids)))
    for i, aa in enumerate(sequence):
        encoding[i, amino_acids.index(aa)] = 1
    return encoding

X= df['Sequence'].apply(one_hot_encode)
seq=[]
for row in X:
    seq.append(row)

X_train, X_test, y_train, y_test = model_selection.train_test_split(seq, y,
                                    train_size=0.80, test_size=0.20, random_state=4)

X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test, y_test, train_size=0.50, test_size=0.50, random_state=4)

from torch.utils.data import Dataset

X_train = [torch.tensor(arr) for arr in X_train]
X_train = pad_sequence(X_train, batch_first=True)


X_test=[torch.tensor(arr) for arr in X_test]
X_test=pad_sequence(X_test, batch_first=True)

X_val=[torch.tensor(arr) for arr in X_val]
X_val=pad_sequence(X_val, batch_first=True)


y_train=torch.tensor(y_train,  dtype=torch.float32)
y_test=torch.tensor(y_test,  dtype=torch.float32)
y_val=torch.tensor(y_val,  dtype=torch.float32)

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# Create an instance of your custom dataset
trainset = CustomDataset(X_train, y_train)
valset = CustomDataset(X_val, y_val)
testset = CustomDataset(X_test, y_test)

# Dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bts, shuffle=True, num_workers=5)
valloader = torch.utils.data.DataLoader(valset, batch_size=bts, shuffle=False, num_workers=5)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=5)

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=20,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            #self.dropouts.append(dropout_fn(dropout))
        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
            """
            Input x is shape (B, L, d_input)
            """
            x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
            for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
                # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

                z = x
                if self.prenorm:
                    # Prenorm
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)

                # Apply S4 block: we ignore the state input and output
                z, _ = layer(z)

                # Dropout on the output of the S4 block
                z = dropout(z)

                # Residual connection
                x = z + x

                if not self.prenorm:
                    # Postnorm
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)

            x = x.transpose(-1, -2)

            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

            # Decode the outputs
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)

            return x

if __name__ == "__main__":
    run = wandb.init(
    # Set the project where this run will be logged
    project="Structured State Space",
    _service_wait=60,
    dir="home/lxxqiu001/temp",
    # Track hyperparameters and run metadata
    config={
        "steps": steps,
        "epochs": epochs,
        "batch_size": bts,
    })
    m=nn.Sigmoid()
    print("Sigmoid")
    # Model
    print('==> Building model..')
    model = S4Model(
        d_input=d_input,
        d_output=n_classes,
        d_model=512, #set arbitrarily
        n_layers=25, #set arbitrarily
        #dropout=args.dropout,
        #prenorm=args.prenorm,
    )


    def setup_optimizer(model, lr, weight_decay, steps_per_epoch):
        """
        S4 requires a specific optimizer setup.
        The S4 layer (A, B, C, dt) parameters typically
        require a smaller learning rate (typically 0.001), with no weight decay.
        The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
        and weight decay (if desired).
        """

        # All parameters in the model
        all_parameters = list(model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )

        # Create a lr scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps_per_epoch)

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(' | '.join([
                f"Optimizer group {i}",
                f"{len(g['params'])} tensors",
            ] + [f"{k} {v}" for k, v in group_hps.items()]))

        return optimizer, scheduler

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    accuracy=[]
    training_loss=[]
    validation_loss=[]
    
    validation_f1=[]
    validation_auc=[]

    test_f1=[]
    test_auc=[]

    optimizer, scheduler = setup_optimizer(
        model, lr=0.0015, weight_decay=0.000001, steps_per_epoch=steps
    )

    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:

                targets = targets.float()
                inputs =inputs.float()
                output=model(inputs)
                print(output)
                loss = criterion(m(output), targets)
                t_loss=loss.item()
                
                training_loss.append(t_loss)

                optimizer.zero_grad()

                #Perform backward pass
                loss.backward()
                optimizer.step()
                wandb.log({"training loss": loss})
	    
        print("Training for 1 epoch is over")

        model.eval()

        for inputs, targets in valloader:
                #Perform forward pass
                targets = targets.float()
                inputs =inputs.float()
                output=model(inputs)
                loss = criterion(m(output), targets)

                v_loss=loss.item()
                validation_loss.append(v_loss)
                
                wandb.log({"validation loss": v_loss})
	
        print("finish validation for one epoch")

    model.eval()
    for inputs, targets in testloader:
                #Perform forward pass
                targets= targets.float()
                inputs=inputs.float()
                output=model(inputs)
                predicted_labels= torch.round(torch.sigmoid(output)) #sigmoid produces probabilities that are rounded to 0 or 1
      
                tmetricAuC =MultilabelAUPRC(num_labels=500, average='macro')
                tmetricAuC.update(predicted_labels, targets)
                tAuC=tmetricAuC.compute()
                print("AuC", tAuC.item())
                targets=targets.squeeze()
                predicted_labels = predicted_labels.squeeze()

                tmetricf1=MulticlassF1Score(num_classes=n_classes)
                tmetricf1.update(predicted_labels, targets)
                tf1=tmetricf1.compute()
                print("F1 Score", tf1.item())
                wandb.log({"Test F1 Score": tf1})

                # Log the AUC score
                wandb.log({"Test AuC": tAuC})

	
    print("finish testing")

    torch.save(model.state_dict(), 's4.pth')
    wandb.save('s4_model_state_dict.pth')

