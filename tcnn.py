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

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TBlock(nn.Module):
    def __init__(self,**kwargs):  
        super(TBlock, self).__init__()    

        self.conv_layer1 = nn.utils.weight_norm(nn.Conv1d(kwargs["in_num"],kwargs["out_num"],kwargs["kernel"],kwargs["stride"],kwargs["padding"],kwargs["dilation"]))

        self.conv_layer2 = nn.utils.weight_norm(nn.Conv1d(kwargs["in_num"],kwargs["out_num"],kwargs["kernel"],kwargs["stride"],kwargs["padding"],kwargs["dilation"]))

        self.chomp1 = Chomp1d(kwargs["c_size"])

        self.chomp2 = Chomp1d(kwargs["c_size"])

        self.relu1 = nn.ReLU()

        self.relu2 = nn.ReLU()

        self.dropout1 = nn.Dropout(kwargs["dropout"])

        self.dropout1 = nn.Dropout(kwargs["dropout"])

        self.net = nn.Sequential