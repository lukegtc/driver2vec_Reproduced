import numpy as np
import torch
import pandas as pd
from math import *
import os


cwd = os.getcwd()

DATA_DIR = os.path.join(cwd,'Reproduction_Data')  #Insert name of datapath here

#Select terrain a single terrain, and train
#Variable set is removed from the original data set so tbale 5 can be replicated
TERRAIN_SET = ['highway','suburban','tutorial','urban']  #Constant Terrain Set. Change if variables are added



def dataset_open( terrain,pct_training = 0.8,variables = [], always_remove = ['FOG','FOG_LIGHTS','FRONT_WIPERS','HEAD_LIGHTS','RAIN','REAR_WIPERS','SNOW']):
    if terrain not in TERRAIN_SET:
        print("Incorrect Terrain Selected")
    else:
        # user_dict = {}
        tot_set = []
        i = 1
        variables.extend(always_remove)
        print(variables)
        
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
        train_len = round(tot_len*pct_training)
        test = round(tot_len*(1-pct_training)/2)
        val = tot_len-train_len-test
        return full_tensor[:,:,:train_len],full_tensor[:,:,train_len:(train_len+test)],full_tensor[:,:,(train_len+test):]









