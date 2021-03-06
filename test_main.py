import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from data_toolkit import *
from model_toolkit import *
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
losses = []
epochs = []
setting = 'test' #train or test
training_tensor, eval_tensor, testing_tensor = Data_Processing.dataset_open('highway', add_noise=False, noise_variance=0.1)
training_tensor,eval_tensor,testing_tensor = training_tensor[:tot_drivers,:,:],eval_tensor[:tot_drivers,:,:],testing_tensor[:tot_drivers,:,:]

training_tensor  = dataset_split(training_tensor,20,1) #Splits any tensor into snippets spaced .20 seconds apart for 1 second intervals

len_set = [32 for x in training_tensor] #Hard coded dataset that includes 32 snippets of data
scoring = 0
tot_ds = Driver_Dataset() # Initializes the total dataset 
dataset = tot_ds.dataset_generator() #Creates the dataset
#Initializes the model to be used to train
model = TCN(c_in = 31,
            wavelet = True,
            l_in = input_length,
            out_n = tot_drivers,
            kernel = 7,
            do_rate = 0.1,
            channel_lst=len_set,
            out_wavelet_size = 15)

evaluator1 = Evaluator(triplet_margin = 1.0,
                        triplet_weight = 0.5) #Initializes the evaluator class


predictor1 = Predictor(model = model, 
                        fast_debug =  False) #Initializes the predictor class

optimizer1 = Optimizer(model_params = model.parameters(),
                        dataset_len = 800,
                        learning_rate = 0.0001,
                        weight_decay = 0.00001,
                        lr_step_epoch = 4,
                        lr_gamma = 0.9,
                        batch_size = 100,
                        disp_steps = 10,
                        max_epochs = 100) #Initializes the optimizer class

test1 = DataLoader(dataset=training_tensor[0,:,0,:],
                    batch_size=batch_size,
                    shuffle = (setting == 'train'),
                    num_workers = workers)

def do_test():
        
    if setting == 'test':
        # The following should be the same as for normal evaluation
        cur_step = optimizer1.total_step
        predictor1.start_prediction(dataset['highway'][0]['training'], 'train')
        loader_name = 'test_lgbm'
        data_loaders = dataset['highway'][0]['test']

        predictor_out = predictor1.lgbm_predict(data_loaders,'test')  #Returns attributes of predictor and data_loader

        scalar_results = evaluator1.evaluate(loader_name,optimizer1,predictor_out)

        print(scalar_results)
    else:
        print('Test skipped')

do_test()
if setting == 'train':
    model.train()

    while not optimizer1.completed():
        iteration = 0
        print(iteration)
        for original, positive, negative, target, data_info in dataset['highway'][0]['training']:
            print('TARGET: ', target.shape, target)

            positive = positive.reshape(1,positive.shape[0],positive.shape[1])
            original = original.permute(0, 2, 1)

            positive = positive.permute(0,2,1)
            negative = negative.permute(0,2,1)
            original, positive, negative = gen_wvlt_set(original, positive, negative)


            original = original.to(device)
            target = target.to(device)
            with torch.set_grad_enabled(True):
                predictions, other_info = model(original,positive,negative)


                other_info['data_info'] = data_info
                info_to_evaluate = {'predictions': predictions,
                                    'ground_truth': target,
                                    'other_info': other_info}

                eval_result = evaluator1.evaluate('train', optimizer1, info_to_evaluate)
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

                predictor1.start_prediction(
                    dataset['highway'][0]['training'], 'train')
  
                predictor_out = predictor1.lgbm_predict(
                    dataset['highway'][0]['test'], 'test')

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
        print(scalar_results['train:loss'].detach().numpy())
        print(optimizer1.cur_epoch)
        losses.append(scalar_results['train:loss'].detach().numpy())
        epochs.append(optimizer1.cur_epoch)
        print(epochs)
        print(losses)
        optimizer1.end_epoch()
    print(epochs)
    print(losses)
    plt.plot(epochs, losses)
    plt.ylabel('Loss [-]', fontsize=16)
    plt.xlabel('Epoch [-]', fontsize=16)
    plt.title('Training loss', fontsize=28)

else:
    print('Training skipped.')

# Plot loss function
plt.show