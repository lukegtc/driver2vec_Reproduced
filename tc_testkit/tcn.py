import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

#Used to get rid of excess padding on the right side
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d( n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d( n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # torch.nn.init.xavier_uniform_(self.conv1.weight)
        # torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        if self.downsample is not None:
            # torch.nn.init.xavier_uniform(self.downsample.weight)
            torch.nn.init.kaiming_uniform_(self.downsample.weight)
            

    def forward(self, x):
        
        out = self.net(x)

        if self.downsample is None:
            res = x  
        else: 
            res = self.downsample(x)

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, c_in, channel_lst, kernel=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layer_set = []
        num_levels = len(channel_lst)
        for i in range(num_levels):
            dilation_size = 2 ** i
            if i == 0:
                in_channels = c_in  
            else: 
                in_channels = channel_lst[i - 1]

            out_channels = channel_lst[i]
            layer_set += [TemporalBlock(in_channels,out_channels,kernel,stride=1,dilation=dilation_size, padding=(kernel - 1) * dilation_size,dropout=dropout)]

        self.network = nn.Sequential(*layer_set)

    def forward(self, x):
        return self.network(x)
