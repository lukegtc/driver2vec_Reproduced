import torch
import numpy as np
from collections import defaultdict

from tqdm import tqdm
import lightgbm as lgb
from utils import *

# from constants import NUM_DRIVERS


def recursive_append(target_dict, source_dict):
    for e in source_dict:
        if type(source_dict) == dict:

            target_dict[e].append(source_dict[e])
    # print('target: ', target_dict.keys())
    return target_dict

def recursive_concat(source_dict):
    for e in source_dict:
        if type(source_dict[e]) == dict or type(source_dict[e]) == defaultdict:
            source_dict[e] = recursive_concat(source_dict[e])
        elif source_dict[e] is not None and e != 'mask':

            source_dict[e] = np.concatenate(source_dict[e])
    
    return source_dict

class Predictor(object):
    """Predictor class for a single model."""

    def __init__(self, model, device, fast_debug):
        self.model = model
        self.device = device
        self.fast_debug = fast_debug
        # self.logger = unique_log_val
        self.num_drivers = 5
        self.train_results = None

        # TODO Move this to constants or to command line
        # Maybe easier to have a spec file for LightGBM
        # https://stackoverflow.com/questions/47370240/multiclass-classification-with-lightgbm
        # https://lightgbm.readthedocs.io/en/latest/Python-Intro.html#training
        self.lgb_param = {'num_leaves': 31,
                          'num_trees': 100,
                          'boosting_type': 'gbdt',
                          'objective': 'multiclass',
                          'num_class': self.num_drivers,
                          'max_depth': 12,
                          'verbosity': 0,
                          'task': 'train',
                          'metric': 'multi_logloss',
                          "learning_rate" : 1e-2,
                          "bagging_fraction" : 0.9,  # subsample
                          "bagging_freq" : 5,        # subsample_freq
                          "bagging_seed" : 341,
                          "feature_fraction" : 0.8,  # colsample_bytree
                          "feature_fraction_seed":341,}
        self.lgb_num_rounds = 15

    def _predict(self, loader, setting, ratio=1, need_triplet_emb=True):
        
        self.model.eval()

        outputs = []
        ground_truth = []
        other_info = defaultdict(list)
        debug_counter = 0

        if setting == 'test':
            orig_features, pos_features, neg_features, targets, data_info = loader

            pos_features = pos_features.reshape(1, pos_features.shape[0], pos_features.shape[1])
            orig_features = orig_features.permute(0, 2, 1)
            pos_features = pos_features.permute(0, 2, 1)
            neg_features = neg_features.permute(0, 2, 1)
            orig_features, pos_features, neg_features = gen_wvlt_set(orig_features, pos_features, neg_features)

            with torch.no_grad():
                predictions, info = self.model(orig_features,
                                               pos_features,
                                               neg_features)  # ,need_triplet_emb)

            outputs.append(predictions)

            ground_truth.append(targets)
            # embeddings.append(emb.cpu())
            # print('OTHER INFO test: ', other_info)
            other_info = recursive_append(other_info, info)
            # print('OTHER INFO test: ', other_info)
            if 'data_info' not in other_info:
                other_info['data_info'] = recursive_append(
                    defaultdict(list),
                    data_info)
            else:
                other_info['data_info'] = recursive_append(
                    other_info['data_info'],
                    data_info)


        elif setting == 'train':
            with tqdm(total=len(loader)) as progress_bar:
                for i in loader:
                    print(len(i))
                    orig_features, pos_features, neg_features, targets, data_info = i
                    if np.random.rand() > ratio:
                        progress_bar.update(targets.size(0))
                        continue
                    pos_features = pos_features.reshape(1, pos_features.shape[0], pos_features.shape[1])
                    orig_features = orig_features.permute(0, 2, 1)
                    pos_features = pos_features.permute(0, 2, 1)
                    neg_features = neg_features.permute(0, 2, 1)
                    orig_features, pos_features, neg_features = gen_wvlt_set(orig_features, pos_features, neg_features)

                    with torch.no_grad():
                        predictions, info = self.model(orig_features,
                                                      pos_features,
                                                      neg_features)   #,need_triplet_emb)
                    outputs.append(predictions)
                    ground_truth.append(targets)
                    # embeddings.append(emb.cpu())
                    other_info = recursive_append(other_info, info)
                    if 'data_info' not in other_info:
                        other_info['data_info'] = recursive_append(
                            defaultdict(list),
                            data_info)
                    else:
                        other_info['data_info'] = recursive_append(
                            other_info['data_info'],
                            data_info)
                    progress_bar.update(targets.size(0))
                    debug_counter += 1
                    if self.fast_debug and debug_counter >= 4:
                        break

        outputs = np.concatenate(outputs)
        
        ground_truth = np.concatenate(ground_truth)
        other_info = recursive_concat(other_info)

        self.model.train()
        return outputs, ground_truth, other_info

    def start_prediction(self, train_loader, setting):

        train_out, train_gt, train_emb = self._predict(train_loader, setting)
        self.train_results = train_out, train_gt, train_emb



    def lgbm_predict(self, other_loader, setting):
        other_out, other_gt, other_info = self._predict(other_loader, setting)
        train_out, train_gt, train_emb = self.train_results

        train_data = lgb.Dataset(train_emb['orig'], label=train_gt)
        bst = lgb.train(self.lgb_param, train_data, self.lgb_num_rounds)
        other_bst_out = bst.predict(other_info['orig'])

        return {'predictions': other_bst_out,
                'ground_truth': other_gt,
                'other_info': other_info}
    
    def end_prediction(self):
        self.train_results = None