import torch
from torch import nn as nn
from torch.nn import functional as F


class RNN(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, nin, nout, es=16, hs=16, nl=3, device=0):
        super().__init__()
        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nout)
        self.to(device)

    def forward(self, x, hidden=None):
        x = F.relu(self.embedding(x))
        x, hidden = self.gru(x, hidden)
        x = self.output_layer(x)
        return x
