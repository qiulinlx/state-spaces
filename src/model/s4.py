import torch.nn as nn
import torch.optim as optim
import torch

# Import S4 model components
from models.s4.s4 import S4Block as S4
from models.s4.s4d import S4D


class S4Model(nn.Module):
    """
    S4 Model for sequence modeling tasks.

    Args:
        d_input (int): Input dimension (e.g., number of features per time step).
        d_output (int): Output dimension (e.g., number of classes).
        d_model (int): Hidden dimension of the model.
        n_layers (int): Number of S4 layers.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        prenorm (bool, optional): Whether to use pre-normalization. Defaults to False.
    """

    def __init__(self, d_input, d_output, d_model, n_layers, dropout=0.2, prenorm=False):
        super().__init__()
        self.prenorm = prenorm

        # Linear encoder to project input to hidden dimension
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))

        # Linear decoder to project hidden dimension to output
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Forward pass for the S4 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, d_output).
        """
        # Project input to hidden dimension
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        # Transpose for S4 layers: (B, L, d_model) -> (B, d_model, L)
        x = x.transpose(-1, -2)

        # Apply S4 layers with residual connections
        for layer, norm in zip(self.s4_layers, self.norms):
            z = x
            if self.prenorm:
                # Pre-normalization
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block
            z, _ = layer(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Post-normalization
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        # Transpose back: (B, d_model, L) -> (B, L, d_model)
        x = x.transpose(-1, -2)

        # Average pooling over the sequence length
        x = x.mean(dim=1)  # (B, L, d_model) -> (B, d_model)

        # Decode to output dimension
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


def setup_optimizer(model, lr, weight_decay, steps_per_epoch):
    """
    Set up the optimizer and learning rate scheduler for the S4 model.

    Args:
        model (nn.Module): The model to optimize.
        lr (float): Base learning rate.
        weight_decay (float): Weight decay for regularization.
        steps_per_epoch (int): Number of steps per epoch for the scheduler.

    Returns:
        optimizer (torch.optim.Optimizer): Configured optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    """
    # Collect all parameters in the model
    all_parameters = list(model.parameters())

    # Separate parameters with and without special optimization settings
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    # Create a cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps_per_epoch)

    return optimizer, scheduler
