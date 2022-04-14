import sys
sys.path.append('.')

import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn

from utils import *


class Optimizer():

    def __init__(self, model_params, dataset_len,learning_rate,weight_decay,lr_step_epoch,lr_gamma,batch_size,disp_steps,max_epochs):
        """
        Args:
            model_params (Parameters): Dictionary of parameters required for the Adam Optimizer
            dataset_len (int): Length of the total dataset
            learning_rate (float): Value that sets the learning rate
            weight_decay (float): Value that sets the decay of the weights for the Adam Optimizer
            lr_step_epoch (int): Important for the step size scheduler within the learning rate function. Adjusts the length of lr decay
            lr_gamma (float): Value that is a multiplier of the lr decay
            batch_size (int): Length of a single batch.
            disp_steps (int): Current step of the epoch.
            max_epochs (int): Total number of epochs
    
        """
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_step_epoch = lr_step_epoch
        self.lr_gamma = lr_gamma
        self.batch_size = batch_size
        self.disp_steps = disp_steps
        self.max_epochs = max_epochs
        # Using Adam
        self.optimizer = torch.optim.Adam(
            model_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.lr_step_epoch,
            gamma=self.lr_gamma)
        self.model_params = model_params

        self.steps_per_epoch = dataset_len // self.batch_size

        self.cur_epoch = 1
        self.epoch_step = 1
        self.total_step = 1

        self.prev_time = time.time()
        self.train_time = []

    def generate_state_dict(self):
        """
        Creates the dictionary that holds all the states
        Returns:
            state_dict (dict[int]): Dictionary containing important values to be carried over into
                                    future epochs.
        """
        # TODO handle LR scheduler
        state_dict = {}
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        state_dict['cur_epoch'] = self.cur_epoch
        state_dict['epoch_step'] = self.epoch_step
        state_dict['total_step'] = self.total_step

        return state_dict

    def load_state_dict(self, state_dict, learning_rate,lr_gamma,lr_step_epoch):
        """
        Initializes the values taken from the state dictionary and assigns them within the class
        Args:
                state_dict (dict[int]): See function generate_state_dict
                learning_rate (float): Learning rate for the optimizer
                lr_gamma (float): Value that is a multiplier of the lr decay
                lr_step_epoch (int): Important for the step size scheduler within the learning rate function. Adjusts the length of lr decay

        """
        # For restarting at new learning rate
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        print(f'Setting new learning rate to {self.learning_rate}')
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate
        
        # For re-setting learning rate scheduler
        if 'gamma' in state_dict['scheduler_state_dict']:
            state_dict['scheduler_state_dict']['gamma'] = \
                lr_gamma
        if 'step_size' in state_dict['scheduler_state_dict']:
            state_dict['scheduler_state_dict']['step_size'] = \
                lr_step_epoch                
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        # Things intrinsic to our own optimizer
        self.cur_epoch = state_dict['cur_epoch']
        self.epoch_step = state_dict['epoch_step']
        self.total_step = state_dict['total_step']


    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def record_train(self):
        cur_time = time.time()
        self.train_time.append(cur_time - self.prev_time)
        self.prev_time = cur_time

    def print_train_status(self):
        if self.total_step % self.disp_steps == 0:
            avg_time = round(np.sum(self.train_time) / len(self.train_time), 2)
            print(f'Epoch {self.cur_epoch}.\t'
                  f'Epoch Step {self.epoch_step}/{self.steps_per_epoch}.\t'
                  f'Total Step {self.total_step}\t'
                  f'Avg Time {avg_time}')
            self.train_time = []

    def end_iter(self):
        self.record_train()
        self.print_train_status()
        self.epoch_step += 1
        self.total_step += 1

    def end_epoch(self):
        self.scheduler.step()
        self.cur_epoch += 1
        self.epoch_step = 1

    def completed(self):
        return self.cur_epoch >= self.max_epochs
