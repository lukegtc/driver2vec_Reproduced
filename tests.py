import torch
import torch.nn as nn
from math import *
import numpy as np


input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)


print(input.shape)
print(target.shape)