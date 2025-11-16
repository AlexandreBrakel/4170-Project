import torch
from torch import nn

class MLP(nn.Module):

    def __init__(self, input_size: float, output_size: float, hidden_size: float, num_hidden_layers: float, dropout_rate: float = 0.2, use_batch_norm: bool = True) :
        super(MLP, self).__init__()
        activation = nn.ReLU()

        # Build network with batch normalization and dropout
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(activation)
        layers.append(nn.Dropout(dropout_rate))
        
        # Additional hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (no batch norm or dropout)
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
