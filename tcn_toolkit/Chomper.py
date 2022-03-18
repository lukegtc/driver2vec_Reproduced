import torch
import torch.nn as nn
from math import *
import numpy as np
# Chomp1D:
#removes excess right padding from Convolutional output https://stats.stackexchange.com/questions/403281/why-chomp-in-temporal-convolutional-network

#https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, input):
        return input[:, :, :-self.chomp_size].contiguous()
