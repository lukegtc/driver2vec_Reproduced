import torch
from torch import nn


class FCN(nn.Module):
    def __init__(self, in_features: int, input_channels: int, hidden_dim: int, out_features: int):
        super(FCN, self).__init__()

        print(in_features, input_channels, in_features*input_channels)
        self.layers1 = nn.Sequential(nn.Linear(in_features * input_channels, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, out_features * 2),
                                     nn.Sigmoid())

        self.layers2 = nn.Sequential(nn.Linear(out_features * 2, out_features))
        print(nn.Linear(in_features * input_channels, hidden_dim).weight.size())
        print(nn.Linear(in_features * input_channels, hidden_dim).bias.size())

    def forward(self, x):
        # x = x.contiguous().view(x.size()[0], -1)
        h = self.layers1(x)
        y = self.layers2(h)
        return y, {'orig': h, 'pos': None, 'neg:': None}