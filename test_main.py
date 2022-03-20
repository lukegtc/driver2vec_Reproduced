from data_toolkit import *
from tcn_toolkit import *


input_tensor = Data_Processing.dataset_open('highway')

print(input_tensor.shape)
len_set = [len(x) for x in input_tensor]
test_net = Full_TCN_wavelet.FullTCNet(31,len_set,7,0.1)
test_wavelet = TCN_wavelet(in_num = ,wavelet,in_len,out_len,kernel,dropout,tot_channels,wave_out_len ) #Fix based on kwargs list in the FTCN module


