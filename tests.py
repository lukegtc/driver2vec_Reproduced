import torch
import torch.nn as nn
from math import *
import numpy as np
a = torch.arange(10).reshape(5,2)
print(a)

print(torch.split(a,5//2,dim=1))