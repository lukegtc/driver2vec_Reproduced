import copy
import re
import multiprocessing
# from utils import *
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def detach_tensor(tensor):
    if type(tensor) != np.ndarray:
        if type(tensor) == list:
            return np.ndarray(tensor)
        else:
            return tensor.cpu().detach().numpy()
    return tensor

def cash_to_tensor(tensor):
    if type(tensor) == np.ndarray:
        return torch.Tensor(tensor)
    return tensor
# https://github.com/adambielski/siamese-triplet/blob/master/losses.py
def loss_triplet_wrapper(loss_inputs):
    # Verify triplet loss
    margin = loss_inputs['triplet_margin']
    weight = loss_inputs['triplet_weight']
    pred = cash_to_tensor(loss_inputs['predictions'])
    target = cash_to_tensor(loss_inputs['ground_truth'])
    orig = cash_to_tensor(loss_inputs['other_info']['orig'])
    pos = cash_to_tensor(loss_inputs['other_info']['pos'])
    neg = cash_to_tensor(loss_inputs['other_info']['neg'])
   
    # # This is to verify magnitude of embeddings
    loss_embedd = orig.norm(2) + pos.norm(2) + neg.norm(2)
    print(f'Triplet embedding magnitude loss {loss_embedd}')

    # Check current driver
    # for i in np.arange(5):
    #     if sum(target[:,i]) == 1.0:
    #         idx = i
    idx = target.tolist().index(1)
    orig = torch.reshape(orig[idx,:], (1,62))
    loss_set = []

    for set in neg:
        set = torch.reshape(set, (1,62))


        losses = F.triplet_margin_loss(orig, pos, neg, margin) * weight + \
                    nn.CrossEntropyLoss()(pred, target.long()) * (1 - weight)
    return losses


pool = multiprocessing.Pool(4)
# We might want multi-target stuff
class Evaluator():
    def __init__(self, device,heavy_log_steps,triplet_margin,loss_fn_name,triplet_weight ):

        self.device = device
        self.heavy_log_steps = heavy_log_steps
    
        # margin for triplet loss
        self.triplet_margin = triplet_margin
        self.define_loss(loss_fn_name)
        self.triplet_weight = triplet_weight

    # Loss related specifications
    def define_loss(self, loss_fn_name):
        if loss_fn_name == 'triplet':
            self.loss_fn = loss_triplet_wrapper
        # elif loss_fn_name == 'cross_entropy':
        #     self.loss_fn = loss_cross_entropy_wrapper
        else:
            raise NotImplementedError(
                f'Loss function {loss_fn_name} not implemented.')

    def loss_backward(self, model, clipping_value):
        self.loss.backward()
        
        gradients = []
        weights = []
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.cpu().detach().numpy()
                pp = p.cpu().detach().numpy()
                gradients.append(g.flatten())
                weights.append(pp.flatten())
        gradient_norm = np.linalg.norm(np.concatenate(gradients, axis=0))
        weight_norm = np.linalg.norm(np.concatenate(weights, axis=0))

        # Some arbitrary limit but gradient should not be this big anyways
        # Basically, skip this step if gradient is crazy. This avoids
        # seeing NaN in model
        # Maybe add NaN check?
        if (gradient_norm > 50 * clipping_value or
            np.isnan(gradient_norm)):
            large_gradient = True
        else:
            large_gradient = False

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        return gradient_norm, weight_norm, large_gradient
    
    def evaluate(self, mode, optimizer, info_for_eval): #, eval_metrics

        info_for_eval['triplet_margin'] = self.triplet_margin
        info_for_eval['triplet_weight'] = self.triplet_weight

        # Evaluations
        scalar_results = {}

        # Loss Function
        loss = self.loss_fn(info_for_eval)
        scalar_results[f'{mode}:loss'] = loss
        
        if mode == 'train':
            # Only keep loss when it is training
            self.loss = loss

            # Learning rate
            for pg in optimizer.optimizer.param_groups:
                learning_rate = pg['lr']
                scalar_results[f'{mode}:learning_rate'] = learning_rate

        return scalar_results
            
      