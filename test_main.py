from data_toolkit import *
from tcn_toolkit import *
from constants import *
from tc_testkit import *
tot_drivers = 5
pct_training = 0.8
training_tensor,eval_tensor,testing_tensor = Data_Processing.dataset_open('highway')
training_tensor,eval_tensor,testing_tensor = training_tensor[:tot_drivers,:,:],eval_tensor[:tot_drivers,:,:],testing_tensor[:tot_drivers,:,:]
# print(training_tensor.shape)
# print(eval_tensor.shape)
# print(testing_tensor.shape)
len_set = [len(x) for x in training_tensor]
# print(len_set)
# test_net = Full_TCN_wavelet.FullTCNet(31,len_set,7,0.1)
# test_wavelet = TCN_wavelet(in_num = 31 ,wavelet = False,in_len = 1000,out_len = 5,kernel = 7,dropout=0.1,tot_channels='25,25,25,25,25,25,25,25',wave_out_len= 15) #Fix based on kwargs list in the FTCN module

# # print(test_wavelet.forward(input = input_tensor))
# print('testing')
# unit_test = TUnit(in_num = 31,out_num = 31,kernel =7,stride = 1,dilation = 1,padding = 6,dropout = 0.1)
# print('testing')
# print(unit_test.forward(input_tensor))
tcn_test = TCN(c_in = 31,wavelet = True, l_in = 800,  out_n = tot_drivers, kernel = 7, do_rate = 0.1, channel_lst=len_set, out_wavelet_size = 15)
print(tcn_test.forward(training_tensor.float()))