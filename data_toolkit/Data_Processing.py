import numpy as np
import torch
import pandas as pd
from math import *
import os
from torch.utils.data import Dataset
from constants import *
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
            print(noise_variance*torch.randn(train_tensor.shape[0], train_tensor.shape[1], train_tensor.shape[2])[0, 0, :10])

        return train_tensor, eval_tensor, test_tensor

def dataset_split(dataset,interval,t_len):
    
    dataset = dataset.numpy()
    print(dataset.shape)
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


class Driver_Dataset(Dataset):
    def __init__(self,dataset):
        super(Driver_Dataset, self).__init__()
        self.old_dataset = dataset  #Tensor
        self.new_dataset = defaultdict(dict)
    def dataset_generator(self):
        for i in range(NUM_DRIVERS):
            for j in TERRAIN_SET:
                training,eval,test = dataset_open(j)
                self.new_dataset['training'].append({'driver_index': i,'terrain/area':j, 'data': training})
                self.new_dataset['eval'].append({'driver_index': i,'terrain/area':j, 'data': eval})
                self.new_dataset['test'].append({'driver_index': i,'terrain/area':j, 'data': test})
        return self.new_dataset