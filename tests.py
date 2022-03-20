import torch
import torch.nn as nn
from math import *
import numpy as np
from .tcn_toolkit import *

test_array = np.array([[i for i in range(10)],[i for i in range(10)]])

print(test_array[:,:-len(test_array[0])])