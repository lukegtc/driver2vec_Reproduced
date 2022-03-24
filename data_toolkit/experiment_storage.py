import numpy as np
import torch
import pandas as pd
from math import *
import os
from logger import *
import datetime
from pytz import timezone

from pathlib import Path
experiments_file = 'experiments'

timezone = timezone('Europe/Amsterdam')
date = datetime.datetime.now(timezone).strftime("%Y-%m-%d")
save_dir = os.path.join(experiments_file,date)
def get_experiment_number(save_dir, experiment_name):
    """Parse directory to count the previous copies of an experiment."""
    dir_structure = os.listdir(save_dir)
    dirnames = [exp_dir.split('/')[-1] for exp_dir in dir_structure[1]]

    ret = 1
    for d in dirnames:
        if d[:d.rfind('_')] == experiment_name:
            ret = max(ret, int(d[d.rfind('_') + 1:]) + 1)
    return ret