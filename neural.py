import torch
from torch import nn

class MLP(nn.Module):

    def __init__(self, input_size: float, output_size: float, hidden_size: float, num_hidden_layers: float) :
        super(MLP, self).__init__()
        activation = nn.ReLU()

        # TODO add batch normalization and/or dropout
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), activation)

        for _ in range(num_hidden_layers):
            self.network.extend(nn.Sequential(nn.Linear(hidden_size, hidden_size), activation))

        self.network.extend(nn.Sequential(nn.Linear(hidden_size, output_size)))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
