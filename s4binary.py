import pandas as pd
import numpy as np
from sklearn import model_selection
import torch.nn as nn
from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
import torch.optim as optim
import torch
from torch.nn.utils.rnn import pad_sequence
from torcheval.metrics import BinaryF1Score, BinaryAUPRC
import wandb

bts=2 #Batch size
n_classes=1
d_input=21 #Length of each vector in sequence
d_output = 1
steps =50
n_epochs=10

print('Imported packages successfully')

df=pd.read_csv('Binartset.csv')

y = df['0']
num_rows = df.shape[0] #no. samples in dataset

#One-hot encoding

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

y_train = y_train.values.reshape(-1)
y_test = y_test.values.reshape(-1)
y_val = y_val.values.reshape(-1)

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
        n_layers=10,
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

    wandb.login(key='7b95dbe82c6138a403e12795e0fd55461555b0e4')
    run = wandb.init(
        # Set the project where this run will be logged
        project="Structured State Space 2",
        dir="home/lxxqiu001/temp",
        # Track hyperparameters and run metadata
        config={
            "steps": steps,
            "epochs": n_epochs,
            "batch_size": bts,
        })


    # Model
    print('==> Building model..')
    model = S4Model(
        d_input=d_input,
        d_output=n_classes,
        d_model=512, #set arbitrarily
        n_layers=10, #set arbitrarily
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


    criterion = nn.BCELoss()
    accuracy=[]
    training_loss=[]
    validation_loss=[]
    
    validation_f1=[]
    validation_auc=[]

    test_f1=[]
    test_auc=[]

    optimizer, scheduler = setup_optimizer(
        model, lr=0.001, weight_decay=0.0000001, steps_per_epoch=steps
    )

    for epoch in range(n_epochs):
        model.train()
        for inputs, targets in trainloader:

                targets = targets.float()
                inputs =inputs.float()
                output=model(inputs)
                targets = torch.reshape(targets, (bts, 1))
                print(output, targets)
                m = nn.Sigmoid()
                loss = criterion(m(output), targets)
                t_loss=loss.item()
                
                training_loss.append(t_loss)

                optimizer.zero_grad()

                #Perform backward pass
                loss.backward()
                print(t_loss)
                optimizer.step()
                wandb.log({"training loss": t_loss})
	    
        print("Training for 1 epoch is over")

        model.eval()

        for inputs, targets in valloader:
                targets = targets.float()
                inputs =inputs.float()
                output=model(inputs)
                psize = output.size()

                # Get the first element as an int
                bsize = psize[0]
                targets = torch.reshape(targets, (bsize, 1))

                m = nn.Sigmoid()
                loss = criterion(m(output), targets)

                v_loss=loss.item()
                validation_loss.append(v_loss)
                print(v_loss)
                wandb.log({"validation loss": v_loss})
	
        print("finish validation for one epoch")
    # torch.save(model, 'bins4.pth')

    # model = S4Model(
    #     d_input=d_input,
    #     d_output=n_classes,
    #     d_model=512, #set arbitrarily
    #     n_layers=10, #set arbitrarily
    #     #dropout=args.dropout,
    #     #prenorm=args.prenorm,
    # )
    # model.load_state_dict(model.state_dict(), torch.load('bins4.pth'))
    # model.eval()
    for inputs, targets in testloader:
                #Perform forward pass
                targets= targets.float()
                targets= targets.view(1, 1)
                inputs=inputs.float()
                output=model(inputs)

                predicted_labels= torch.round(torch.sigmoid(output)) #sigmoid produces probabilities that are rounded to 0 or 1
                print(predicted_labels, targets)
                tmetricAuC =BinaryAUPRC()
                tmetricAuC.update(predicted_labels, targets)
                tAuC=tmetricAuC.compute()
                AuC=(tAuC.item())
                print("TauPRC",AuC)
                targets = targets.view(-1)
                predicted_labels = predicted_labels.view(-1)

                tmetricf1=BinaryF1Score()
                tmetricf1.update(predicted_labels, targets)
                tf1=tmetricf1.compute()
                f1=(tf1.item())
                print("F1",f1)
                wandb.log({"Test F1 Score": f1})

                # Log the AUC score
                wandb.log({"Test AuC": AuC})

	
    print("finish testing")

    torch.save(model.state_dict(), 'bins4.pth')
    wandb.save('s4_model_state_dict.pth')

