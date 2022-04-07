import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader


from data_toolkit import *
from model_toolkit import *
from tcn_toolkit import *
from utils import *
from tc_testkit import *
from model_toolkit import *
tot_drivers = 5
pct_training = 0.8
device = 'cpu'
do_eval = True
workers = 4
save_steps = 800
input_length = 100
eval_steps = 400
fast_debug = False
clipping_value = 1.0
batch_size = 36  #384
setting = 'test' #train or test
training_tensor, eval_tensor, testing_tensor = Data_Processing.dataset_open('highway', add_noise=False, noise_variance=0.1)
training_tensor,eval_tensor,testing_tensor = training_tensor[:tot_drivers,:,:],eval_tensor[:tot_drivers,:,:],testing_tensor[:tot_drivers,:,:]

training_tensor  = dataset_split(training_tensor,20,1) #Splits any tensor into snippets spaced .20 seconds apart for 1 second intervals

len_set = [32 for x in training_tensor]
scoring = 0
tot_ds = Driver_Dataset()
dataset = tot_ds.dataset_generator()

model = TCN(c_in = 31,wavelet = True, l_in = input_length,  out_n = tot_drivers, kernel = 7, do_rate = 0.1, channel_lst=len_set, out_wavelet_size = 15)



#TODO: column selector


evaluator1 = Evaluator('cuda',100,1.0,'triplet',0.5 )

eval_metrics = TRIPLET_EVAL_METRICS

predictor1 = Predictor(model, 'cuda', False)

optimizer1 = Optimizer(model.parameters(),800,0.0001,0.00001,4,0.9,100,10,100)
test1 = DataLoader(dataset=training_tensor[0,:,0,:],batch_size=batch_size,shuffle = (setting == 'train'), num_workers = workers)


def do_test():
        
    if setting == 'test':
        # The following should be the same as for normal evaluation
        cur_step = optimizer1.total_step

        predictor1.start_prediction(dataset['highway'][0]['training'])
        loader_name = 'test_lgbm'
        data_loaders = dataset['highway'][0]['test']
        # TODO: CHANGE
        predictor_out = predictor1.lgbm_predict(loader_name,data_loaders,'lgbm_predict')  #Returns attributes of predictor and data_loader
        # TODO: CHANGE
        scalar_results = evaluator1.evaluate(loader_name,optimizer1,predictor_out,data_loaders)
        print(scalar_results)
    else:
        print('Test skipped')

do_test()
if setting == 'train':
    model.train()

    while not optimizer1.completed():
        iteration = 0
        for original, positive, negative, target, data_info in dataset['highway'][0]['training']:
            # original = original.permute(0, 2, 1)
            positive = positive.reshape(1,positive.shape[0],positive.shape[1])
            original = original.permute(0, 2, 1)

            positive = positive.permute(0,2,1)
            negative = negative.permute(0,2,1)

            original = torch.Tensor(gen_wavelet(np.array(original,dtype=np.float32)))
            negative = torch.Tensor(gen_wavelet(np.array(negative,dtype=np.float32)))
            positive = torch.Tensor(gen_wavelet(np.array(positive,dtype=np.float32)))
            

            original = original.to(device)
            target = target.to(device)
            with torch.set_grad_enabled(True):
                predictions, other_info = model(original,positive,negative)


                other_info['data_info'] = data_info
                info_to_evaluate = {'predictions': predictions,
                                    'ground_truth': target,
                                    'other_info': other_info}

                eval_result = evaluator1.evaluate('train', optimizer1, info_to_evaluate) #,eval_metrics['train']['train']
                scalar_results = eval_result
  

            optimizer1.zero_grad()
            # Compute gradient norm to prevent gradient explosion
            gradient_norm, weight_norm, large_gradient = evaluator1.loss_backward(model,clipping_value)
            scalar_results[f'train:gradient_norm'] = gradient_norm
            scalar_results[f'train:weight_norm'] = weight_norm
            print('scalar results: ',scalar_results)

            # If gradient is large, just don't step that one
            if not large_gradient:
                print('optimising step')
                optimizer1.step()

            optimizer1.end_iter()
            print(iteration)
            iteration += 1
            if (fast_debug or optimizer1.total_step == 50 or(do_eval and optimizer1.total_step % eval_steps == 0)):

                cur_step = optimizer1.total_step
                #TODO: FIX
                
                predictor1.start_prediction(dataset['highway'][0]['training'])
                #TODO: FIX
                predictor_out = predictor1.lgbm_predict(
                    dataset['highway'][0]['testing'],
                    'lgbm_predict')
                #TODO: FIX
                eval_result = evaluator1.evaluate(
                    'test_lgbm',
                    optimizer1,
                    predictor_out) #,scoring)
                scalar_results = eval_result
            
            if (fast_debug or (optimizer1.total_step % save_steps == 0)):
                print({'save:steps': f'Training progress saved at '
                                               f'step {optimizer1.total_step}'},
                                optimizer1.total_step)

                do_test()

        optimizer1.end_epoch()
else:
    print('Training skipped.')