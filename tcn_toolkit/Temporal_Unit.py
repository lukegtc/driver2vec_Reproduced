import torch
import torch.nn as nn

# from .Chomper import Chomp1d

# Chomp1D:
#removes excess right padding from Convolutional output https://stats.stackexchange.com/questions/403281/why-chomp-in-temporal-convolutional-network

#https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, input):
        return input[:, :, :-self.chomp_size].contiguous()


class TUnit(nn.Module):
    def __init__(self,**kwargs):
        super(TUnit, self).__init__()    
        
        print('check3')

        self.conv_layer1 = nn.utils.weight_norm(nn.Conv1d(kwargs["in_num"],kwargs["out_num"],kwargs["kernel"],kwargs["stride"],kwargs["padding"],kwargs["dilation"]))

        self.conv_layer2 = nn.utils.weight_norm(nn.Conv1d(kwargs["out_num"],kwargs["out_num"],kwargs["kernel"],kwargs["stride"],kwargs["padding"],kwargs["dilation"]))

        self.chomp1 = Chomp1d(kwargs["padding"])

        self.chomp2 = Chomp1d(kwargs["padding"])

        self.relu1 = nn.ReLU()

        self.relu2 = nn.ReLU()

        self.dropout1 = nn.Dropout(kwargs["dropout"])

        self.dropout2 = nn.Dropout(kwargs["dropout"])

        self.net = nn.Sequential(self.conv_layer1,self.conv_layer2,self.chomp1,self.chomp2,self.relu1,self.relu2,self.dropout1,self.dropout2)

        

        if kwargs["in_num"] != kwargs["out_num"]:
            self.downsample = nn.Conv1d(kwargs["in_num"], kwargs["out_num"], 1)
        else:
            self.downsample = None
        self.relu = nn.ReLU()
        self.init_weights()

        #Weight Initialization
    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv_layer1.weight)
        nn.init.kaiming_uniform_(self.conv_layer2.weight)

        if self.downsample != None:
            nn.init.kaiming_uniform_(self.downsample.weight)

        #Forward Step

        def forward(self,input):
            print('check3')
            output = self.net(input)
            if self.downsample == None:
                result = input
            else:
                self.downsample(input)

            return self.relu(output+result)

