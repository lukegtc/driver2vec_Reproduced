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
# print(training_tensor.shape)
# print(eval_tensor.shape)
# print(testing_tensor.shape)
len_set = [32 for x in training_tensor]

# Fully connected layers
in_features = training_tensor.size(dim=0) * training_tensor.size(dim=2)
input_channels = training_tensor.size(dim=1)
fc_test = FCN(in_features=in_features, input_channels=input_channels, hidden_dim=256, out_features=15)
fc_test.forward(training_tensor)

# print(len_set)
# test_net = Full_TCN_wavelet.FullTCNet(31,len_set,7,0.1)
# test_wavelet = TCN_wavelet(in_num = 31 ,wavelet = False,in_len = 1000,out_len = 5,kernel = 7,dropout=0.1,tot_channels='25,25,25,25,25,25,25,25',wave_out_len= 15) #Fix based on kwargs list in the FTCN module

# # print(test_wavelet.forward(input = input_tensor))
# print('testing')
# unit_test = TUnit(in_num = 31,out_num = 31,kernel =7,stride = 1,dilation = 1,padding = 6,dropout = 0.1)
# print('testing')
# print(unit_test.forward(input_tensor))
model = TCN(c_in = 31,wavelet = True, l_in = 800,  out_n = tot_drivers, kernel = 7, do_rate = 0.1, channel_lst=len_set, out_wavelet_size = 15)
# model.forward(training_tensor.float())


#TODO: column selector


evaluator1 = Evaluator('cuda',100,False,1.0,'triplet',0.5 )

eval_metrics = DATASET_EVAL_METRICS['divided_splitted_normalized']

predictor1 = Predictor(model, 'cuda', False)

optimizer1 = Optimizer(model.parameters(),800,0.0001,0.00001,4,0.9,384,10,100)


def do_test():
        
    if setting == 'test':
        # The following should be the same as for normal evaluation
        cur_step = optimizer1.total_step

        predictor1.start_prediction(training_tensor)
        for loader_name in eval_metrics['test']:
            # TODO: CHANGE
            predictor_out = predictor1.named_predict(
                loader_name,
                data_loaders['test'][loader_name],
                eval_metrics['test'][loader_name][1]
            )
            # TODO: CHANGE
            test_result = evaluator1.evaluate(loader_name,optimizer1,predictor_out,eval_metrics['test'][loader_name][0])
            scalar_results, image_results = test_result
            print(scalar_results)
    else:
        print('Test skipped')


if setting == 'train':
    model.train()

    while not optimizer1.completed():
            # TODO: CHANGE
        for features, pos_features, neg_features, target, data_info in data_loaders['train']['train']:

            # TODO Fix this by getting the right loss function rather than
            # skipping the ones with incorrect shape
            if len(features) == batch_size:
                features = features.to(device)
                target = target.to(device)
                with torch.set_grad_enabled(True):
                    predictions, other_info = model(features,pos_features,neg_features)
                    other_info['data_info'] = data_info
                    info_to_evaluate = {'predictions': predictions,
                                        'ground_truth': target,
                                        'other_info': other_info}

                    eval_result = evaluator1.evaluate('train', optimizer1,info_to_evaluate,eval_metrics['train']['train'])
                    scalar_results, image_results = eval_result

                optimizer1.zero_grad()
                # Compute gradient norm to prevent gradient explosion
                gradient_norm, weight_norm, large_gradient = evaluator1.loss_backward(model,clipping_value)
                scalar_results[f'train:gradient_norm'] = gradient_norm
                scalar_results[f'train:weight_norm'] = weight_norm

                # If gradient is large, just don't step that one
                if not large_gradient:
                    optimizer1.step()
            else:
                print(f'Skipping batch with size {len(features)} '
                           f'at total step {optimizer1.total_step}')
            optimizer1.end_iter()
            
            if (fast_debug or optimizer1.total_step == 50 or(do_eval and optimizer1.total_step % eval_steps == 0)):

                cur_step = optimizer1.total_step
#TODO: FIX
                predictor1.start_prediction(data_loaders['train']['train'])
                for loader_name in eval_metrics['eval']:
                    #TODO: FIX
                    predictor_out = predictor1.named_predict(
                        loader_name,
                        data_loaders['eval'][loader_name],
                        eval_metrics['eval'][loader_name][1]
                    )
#TODO: FIX
                    eval_result = evaluator1.evaluate(
                        loader_name,
                        optimizer1,
                        predictor_out,
                        eval_metrics['eval'][loader_name][0])
                    scalar_results, image_results = eval_result
            
            if (fast_debug or (optimizer1.total_step % save_steps == 0)):
                print({'save:steps': f'Training progress saved at '
                                               f'step {optimizer1.total_step}'},
                                optimizer1.total_step)

                do_test()

        optimizer1.end_epoch()
else:
    print('Training skipped.')