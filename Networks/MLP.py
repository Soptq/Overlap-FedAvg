import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(dim_hidden, dim_hidden)
        self.layer_output = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])  # input layer
        x = self.layer_input(x) # hidden layer 1
        x = self.relu(x) # hidden layer 1 relu
        x = self.layer_hidden(x)
        x = self.relu(x)
        x = self.layer_output(x)
        return x