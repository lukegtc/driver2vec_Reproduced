import pywt
import numpy as np
import torch

def gen_wavelet(data):
    new_set = []
    for j in data:
        
        ncol = j.shape[1]
        nrow = j.shape[0]
        for i in range(ncol):
            cur_col = j[:,i].copy()
            (cA, cD) = pywt.dwt(cur_col, 'haar')
            concat = np.concatenate((cA,cD), 0)
            new_col = np.reshape(concat,(nrow,1))
    
            j = np.hstack((j,new_col))
        new_set.append(j)
    return new_set



def gen_wvlt_set(orig,pos,neg):

    orig_wvlt = torch.Tensor(gen_wavelet(np.array(orig, dtype=np.float32)))
    pos_wvlt = torch.Tensor(gen_wavelet(np.array(pos, dtype=np.float32)))
    neg_wvlt = torch.Tensor(gen_wavelet(np.array(neg, dtype=np.float32)))

    return orig_wvlt, pos_wvlt, neg_wvlt