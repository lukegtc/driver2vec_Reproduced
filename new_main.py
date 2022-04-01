import torch
from torch import nn


from data_toolkit import *
from model_toolkit import *
from tcn_toolkit import *
from constants import *
from tc_testkit import *
from model_toolkit import *

tot_drivers = 5
pct_training = 0.8
device = 'cuda'
do_eval = True
save_steps = 800
eval_steps = 400
fast_debug = False
clipping_value = 1.0
batch_size = 384
setting = 'test' #train or test
training_tensor, eval_tensor, testing_tensor = Data_Processing.dataset_open('highway', add_noise=True, noise_variance=0.1)
training_tensor,eval_tensor,testing_tensor = training_tensor[:tot_drivers,:,:],eval_tensor[:tot_drivers,:,:],testing_tensor[:tot_drivers,:,:]
training_set  = dataset_split(training_tensor,20,1) #Splits any tensor into snippets spaced .20 seconds apart for 1 second intervals
eval_set = dataset_split(eval_tensor,20,1)
test_set = dataset_split(testing_tensor,20,1)
len_set = [31 for x in training_tensor]


#Initialize Model
model = TCN(c_in = 31,wavelet = True, l_in = 800,  out_n = tot_drivers, kernel = 7, do_rate = 0.1, channel_lst=len_set, out_wavelet_size = 15)

#initialize predictor
predictor1 = Predictor(model, 'cuda', False)
#initialize evaluator/scorer
evaluator1 = Evaluator('cuda',100,False,1.0,'triplet',0.5 )
#initialize optimizer
optimizer1 = Optimizer(model.parameters(),800,0.0001,0.00001,4,0.9,384,10,100)


if setting == 'test':
    # The following should be the same as for normal evaluation
    cur_step = optimizer1.total_step

    predictor1.start_prediction(training_tensor)
    loader_name = 'test_lgbm'
    data_loaders = testing_tensor
    # TODO: CHANGE
    predictor_out = predictor1.lgbm_predict(loader_name,data_loaders,'lgbm_predict')  #Returns attributes of predictor and data_loader
    # TODO: CHANGE
    scalar_results, image_results = evaluator1.evaluate(loader_name,optimizer1,predictor_out,eval_metrics['test'][loader_name][0])
    print(scalar_results)
else:
    print('Test skipped')




if setting == 'train':
    model.train()