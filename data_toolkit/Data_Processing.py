import numpy as np
import torch
import pandas as pd
from math import *
import os
from torch.utils.data import Dataset
from collections import defaultdict
from utils import *
cwd = os.getcwd()

DATA_DIR = os.path.join(cwd,'Reproduction_Data')  #Insert name of datapath here

#Select terrain a single terrain, and train
#Variable set is removed from the original data set so tbale 5 can be replicated
 #Constant Terrain Set. Change if variables are added



def dataset_open( terrain,
                  add_noise=False,
                  noise_variance=0,
                  pct_training=0.8,
                  variables = [],
                  always_remove = ['FOG','FOG_LIGHTS','FRONT_WIPERS','HEAD_LIGHTS','RAIN','REAR_WIPERS','SNOW']):

    """
    Function that opens the specified terrain dataset
    Args:
        terrain (str): Type of terrain selected.
        add_noise (bool): Determines whether or not to add noise ot the data set.
        noise_variance (float): Variable that increases or decreases the added noise.
        pct_training (float): Percentage of the total dataset that is used for training.
        variables (list[str]): List of columns that need to be removed. Useful for the ablation process.
        always_remove(list[str]): List of columns that are not useful for the reproduction.
    Returns:
        training_tensor (tensor[float]): The selection of data used to train the model. 
        eval_tensor (tensor[float]): The selection of data used to evaluate the model.
        test_tensor (tensor[float]): The selection of data used to test the model.
    """
    if terrain not in TERRAIN_SET:
        print("Incorrect Terrain Selected")
    else:
        # user_dict = {}
        tot_set = []
        i = 1
        variables.extend(always_remove)
        # print(variables)

        for folder in os.listdir(DATA_DIR):
            
            USER_DIR = os.path.join(DATA_DIR,folder)
            user_set = []
            for file in os.listdir(USER_DIR):
                #select terrain
                if terrain in file:
                    #isolate parameter
                    if variables:
                        df = pd.read_csv(os.path.join(USER_DIR,file))
                        df = df.iloc[:,1:]
                        df = df.loc[:,~df.columns.isin(variables)]
                    else:
                        
                        df = pd.read_csv(os.path.join(USER_DIR,file))
                    # trial_dict = {}
                    
                    for columnName, columnData in df.iteritems():
                        # trial_dict[columnName] = torch.tensor(columnData.values)

                        user_set.append(np.array(columnData))
                    # user_dict[i] =trial_dict 

            tot_set.append(user_set)
            i +=1
        
        full_tensor = torch.tensor(tot_set)
        tot_len = full_tensor.shape[2]
        train_len = round(tot_len * pct_training)
        test_len = round(tot_len * (1 - pct_training) / 2)
        eval_len = tot_len - train_len - test_len

        train_tensor = full_tensor[:,:,:train_len]
        eval_tensor  = full_tensor[:,:,train_len:(train_len+test_len)]
        test_tensor  = full_tensor[:,:,(train_len+test_len):]

        if add_noise:
            train_tensor += noise_variance * torch.randn(train_tensor.shape[0], train_tensor.shape[1], train_tensor.shape[2])
            # print(noise_variance*torch.randn(train_tensor.shape[0], train_tensor.shape[1], train_tensor.shape[2])[0, 0, :10])

        return train_tensor, eval_tensor, test_tensor

def dataset_split(dataset,interval,t_len):
    """
    Args:
        dataset (tensor[float]): Tensor that contains the data that must be split up into intervals
        interval (int): Number of datapoints that each sample interval is shifted by.
        t_len (float): Number of seconds the sample interval is long.
    Returns:
        A tensor of a split dataset of time series.
    """
    dataset = dataset.numpy()
    # print(dataset.shape)
    #interval is the amount of time between the starting points of each sample
    new_set = []
    for driver in dataset:
        # print(driver.shape)
        new_driver_set = []
        for channel in driver:
            # print(channel.shape)
            x = 0
            new_channel_set = []
            while True:
                if t_len*100+interval*x <= len(channel):
   
                    new_channel_set.append(channel[interval*x:t_len*100+interval*x])
                else:
                    break
                x+=1
            new_driver_set.append(new_channel_set)
        new_set.append(new_driver_set)
    new_set = np.array(new_set)
    # print(new_set)

    return torch.Tensor(new_set)


class Driver_Dataset():
    """
    Class that initializes the dataset to be interpreted by the model
    """
    def __init__(self,interval = 20,t_len = 1):
        # super(Driver_Dataset, self).__init__()
          #Tensor
        self.interval = interval
        self.t_len = t_len
        self.new_dataset = {}



    def dataset_generator(self):
        """
            Returns:
                terrain_set (dict[float]): A dictionary containing dictionaries for the training, eval and test sets.
                                           Each of these dictionaries contains required info to be passe don during their
                                           respective phases in teh model.
        """
        terrain_set = {}
        for j in TERRAIN_SET:
                driver_set = {}
                for i in range(NUM_DRIVERS):
                    single_driver_set = {}
                    #Open dataset based on terrain
                    training,eval,test = dataset_open(j)
                    original = dataset_split(training,self.interval,self.t_len) 
                                   
                    #positive is just the one driver you want to evaluate
                    positive = original[i,:,:,:]
                    positive_eval = eval[i,:,:]
                    positive_test = test[i,:,:]
                    #Negative is ALL other drivers
                    negative = torch.Tensor(original.numpy()[np.arange(len(original.numpy()))!=i])
                    negative_eval = torch.Tensor(eval.numpy()[np.arange(len(eval.numpy()))!=i])
                    negative_test = torch.Tensor(test.numpy()[np.arange(len(test.numpy()))!=i])
                    #Target is the index of the driver, nothing special
                    target = np.zeros([5])
                    target[i] = 1
                    data_info = {'mask':0, 'other_gt':[np.arange(len(original.numpy()))!=i]}
                    data_info_eval = {'mask':0, 'other_gt':[np.arange(len(eval.numpy()))!=i]}
                    data_info_test = {'mask':0, 'other_gt':[np.arange(len(test.numpy()))!=i]}
                    driver_segments = []
                    for k in range(original.shape[2]):

                        driver_segments.append([torch.Tensor(original[:,:,k,:]),torch.Tensor(positive[:,k,:]),torch.Tensor(negative[:,:,k,:]),torch.Tensor(target),data_info])
           
                    single_driver_set['training'] = driver_segments
                    single_driver_set['eval'] = [eval,positive_eval,negative_eval,torch.Tensor(target),data_info_eval]
                    single_driver_set['test'] = [test,positive_test,negative_test,torch.Tensor(target),data_info_test]
                    driver_set[i] = single_driver_set
                terrain_set[j] = driver_set

        return terrain_set
        
