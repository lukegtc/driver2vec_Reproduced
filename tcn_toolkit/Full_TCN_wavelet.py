import torch
import torch.nn as nn
from math import *
from .Temporal_Unit import *


class FullTCNet(nn.Module):
    def __init__(self,in_num, channel_num,kernel,dropout):
        super(FullTCNet, self).__init__()
        self.layer_set = []
        #Cycles through Channels
        for i in range(len(channel_num)):
            #sets number of channels equal to the output channels for this particular channel
            out_channels = channel_num[i]

            #sets number of in_channels, where first in channel is the number of inputs
            if i == 0:
                in_channels = in_num
            else:
                in_channels = channel_num[i-1]

            self.layer_set.append(TUnit(in_num = in_channels, out_num = out_channels,kernel= kernel,stride=1,dilation = 2**i,padding = (kernel-1)*2**i,dropout = dropout))
        self.net = nn.Sequential(*self.layer_set)
        

    def forward(self,input):
        print('check1')
        return self.net(input)


class TCN_wavelet(nn.Module):
    def __init__(self, **kwargs):
        super(TCN_wavelet, self).__init__()
        self.in_num = kwargs["in_num"]
        self.wavelet = kwargs['wavelet']
        self.in_len = kwargs['in_len']
        out_len = kwargs['out_len']
        kernel = kwargs['kernel']
        dropout = kwargs['dropout']
        tot_channels = kwargs['tot_channels']   #Check this list of channel names? no
        wave_out_len = kwargs['wave_out_len']
        channel_num  = [int(x) for x in tot_channels.split(',')]
        linear_size = channel_num[-1]

        if self.wavelet:
            self.in_num = self.in_num//2
            wave_size = self.in_len*self.in_num//2
            self.lin_wave = nn.Linear(wave_size, wave_out_len)
            linear_size+=2*wave_out_len

        self.tcnet = FullTCNet(self.in_num,channel_num,kernel,dropout)  
        self.batch_norm_input = nn.BatchNorm1d(linear_size) #Batch normalized input
        self.lin_layer = nn.Linear(linear_size,out_len)  #Linear layer


    def forward(self,**kwargs):
        print('check2')
        if self.wavelet:
            split_tensor = torch.split(kwargs['input'],self.in_num,2)
            input = split_tensor[0]
            wavelet_input_tot = split_tensor[1]
            in_wavelet_split_1 = torch.split(wavelet_input_tot,self.in_len//2, 1)[0]
            in_wavelet_split_2 = torch.split(wavelet_input_tot,self.in_len//2,1)[1]
            b = split_tensor.size()[0]
            out_wavelet_split_1 = self.lin_wave(in_wavelet_split_1.reshape(b,-1,1).squeeze())
            out_wavelet_split_2 = self.lin_wave(in_wavelet_split_2.reshape(b,-1,1).squeeze())

        input = kwargs['input'].permute(0,2,1)

        y1 = self.tcnet(input)  #size = N,C,L
        last_y1 = y1[:,:-1]

        if self.wavelet:
            last_y1 = torch.cat([last_y1,out_wavelet_split_1,out_wavelet_split_2],dim=1)


        norm = self.batch_norm_input(last_y1)

        final = self.lin_layer(norm)
        return final,norm