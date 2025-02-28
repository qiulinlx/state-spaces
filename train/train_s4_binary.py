import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryF1Score, BinaryAUPRC
import wandb

from src.model.s4 import S4Model, setup_optimizer
from src.dataloader import ProteinDataset, load_binary_data
# Import S4 model components
from models.s4.s4 import S4Block as S4
from models.s4.s4d import S4D

# Set up argparse for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an S4 model on protein sequence data.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--steps", type=int, default=50, help="Steps per epoch.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0000001, help="Weight decay for optimizer.")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension.")
    parser.add_argument("--n_layers", type=int, default=10, help="Number of S4 layers.")
    parser.add_argument("--data_path", type=str, default="Binaryset.csv", help="Path to the dataset.")
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API key.")
    return parser.parse_args()

# Training loop
def train(model, trainloader, criterion, optimizer, epoch, wandb):
    model.train()
    for inputs, targets in trainloader:
        targets = targets.float().view(-1, 1)
        inputs = inputs.float()
        outputs = model(inputs)
        loss = criterion(torch.sigmoid(outputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"training_loss": loss.item()})

# Validation loop
def validate(model, valloader, criterion, wandb):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in valloader:
            targets = targets.float().view(-1, 1)
            inputs = inputs.float()
            outputs = model(inputs)
            loss = criterion(torch.sigmoid(outputs), targets)
            total_loss += loss.item()
    wandb.log({"validation_loss": total_loss / len(valloader)})

# Testing loop
def test(model, testloader, wandb):
    model.eval()
    f1_metric = BinaryF1Score()
    auc_metric = BinaryAUPRC()
    with torch.no_grad():
        for inputs, targets in testloader:
            targets = targets.float().view(-1, 1)
            inputs = inputs.float()
            outputs = model(inputs)
            predicted_labels = torch.round(torch.sigmoid(outputs))
            f1_metric.update(predicted_labels, targets)
            auc_metric.update(predicted_labels, targets)
    f1 = f1_metric.compute().item()
    auc = auc_metric.compute().item()
    wandb.log({"test_f1": f1, "test_auc": auc})

# Main function
def main():
    args = parse_args()

    # Initialize WandB
    wandb.login(key=args.wandb_key)
    run = wandb.init(project="Structured State Space 2", config=args)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_binary_data(args.data_path)

    # Create datasets and dataloaders
    trainset = ProteinDataset(X_train, y_train)
    valset = ProteinDataset(X_val, y_val)
    testset = ProteinDataset(X_test, y_test)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=5)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=5)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=5)

    # Initialize model
    model = S4Model(d_input=21, d_output=1, d_model=args.d_model, n_layers=args.n_layers)
    criterion = nn.BCELoss()
    optimizer, scheduler = setup_optimizer(model, args.lr, args.weight_decay, args.steps)

    # Training and validation
    for epoch in range(args.epochs):
        train(model, trainloader, criterion, optimizer, epoch, wandb)
        validate(model, valloader, criterion, wandb)
        scheduler.step()

    # Testing
    test(model, testloader, wandb)

    # Save model
    torch.save(model.state_dict(), "models/binary/s4_binary.pth")
    wandb.save("s4_binary.pth")

if __name__ == "__main__":
    main()