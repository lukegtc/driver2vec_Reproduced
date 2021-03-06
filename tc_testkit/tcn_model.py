import torch.nn.functional as F
from torch import nn
from tc_testkit.tcn import TemporalConvNet
import torch
from typing import List


#Based on N,C,L where N is the number of drivers, C is the number of channels and L is the length of each channel

class TCN(nn.Module):
    def __init__(self, c_in:int,wavelet:bool,l_in:int, out_n:int,kernel:int,do_rate:float,channel_lst:List[int],out_wavelet_size:int):
        super(TCN, self).__init__()
        self.c_in = c_in
        self.wavelet = wavelet
        self.l_in = l_in
        dropout = do_rate
        wavelet_output_size = out_wavelet_size
        # channel_lst  = [int(x) for x in channel_lst.split(',')]
        linear_size = channel_lst[-1]

        if self.wavelet:
            self.c_in = self.c_in//2
            wvlt_size = self.l_in * self.c_in // 2
            self.linear_wavelet = nn.Linear(wvlt_size, wavelet_output_size)
            linear_size += 2 * wavelet_output_size

        self.tcn = TemporalConvNet(self.c_in,channel_lst,kernel,dropout)
        
        self.input_bn = nn.BatchNorm1d(linear_size)
        self.linear = nn.Linear(linear_size, out_n)
        

    def forward_TCN(self, inputs,positive,negative):  #,positive,negative
        """Inputs have to have dimension (N, C_in, L_in)"""
        if self.wavelet:

            splits = torch.split(inputs, self.c_in, dim=2)
           
            inputs = splits[0]
            wvlt_inputs = splits[1]


            splits2 = torch.split(wvlt_inputs,self.l_in // 2,dim=1)

            wvlt_inputs_1 = splits2[0]
            wvlt_inputs_2 = splits2[1]

            bsize = inputs.size()[0]
            wvlt_out1 = self.linear_wavelet(wvlt_inputs_1.reshape(bsize, -1, 1).squeeze())
            wvlt_out2 = self.linear_wavelet(wvlt_inputs_2.reshape(bsize, -1, 1).squeeze())

        inputs = inputs.permute(0, 2, 1)

        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        last = y1[:, :, -1]

        if self.wavelet:

            if wvlt_out1.ndim == 1:

                wvlt_out1 = wvlt_out1.reshape(1,wvlt_out1.shape[0])
                wvlt_out2 = wvlt_out2.reshape(1,wvlt_out2.shape[0])

            last = torch.cat([last, wvlt_out1, wvlt_out2], dim=1)

        if last.shape[0] == 1:
            normalized = last
        else:
            normalized = self.input_bn(last)
        o = self.linear(normalized)

        
        return o, {'orig': normalized, 'pos': positive, 'neg': negative}

    def forward(self,inputs,positive,negative):
        o1, cur_emb_out = self.forward_TCN(inputs, None, None)  # input should have dimension (N, C, L)
        last = nn.Sigmoid()(cur_emb_out['orig'])
        # print(last.shape)
        o, pos_emb_out = self.forward_TCN(positive, None, None)
        o, neg_emb_out = self.forward_TCN(negative, None, None)    
        plast = nn.Sigmoid()(pos_emb_out['orig'])
        nlast = nn.Sigmoid()(neg_emb_out['orig'])      


        return o1, {'orig': last, 'pos': plast, 'neg': nlast}