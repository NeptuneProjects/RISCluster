from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import os
import sys
sys.path.insert(0, '../RISCluster/')

import h5py
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import importlib as imp

import networks
imp.reload(networks)
from  networks import AEC, DCEC

import plotting
imp.reload(plotting)
from plotting import view_DCEC_output as w_spec

import processing
imp.reload(processing)

import production
imp.reload(production)

import utils
imp.reload(utils)

label_list = []

idx1 = np.arange(0,10)
label1 = np.random.choice(np.arange(0,11),10)
other = np.random.rand(10,20)
A = [
        {
        'idx': idx1[i],
        'label': label1[i],
        'other': other
        } for i in range(len(idx1))
    ]

label_list += [{k: v for k, v in d.items() if (k == 'idx' or k == 'label')} for d in A]
print(label_list)
# print('=============')
idx2 = np.arange(10,20)
label2 = np.random.choice(np.arange(0,11),10)
B = [
        {
        'idx': idx2[i],
        'label': label2[i],
        'other': other
        } for i in range(len(idx2))
    ]

label_list += [{k: v for k, v in d.items() if (k == 'idx' or k == 'label')} for d in B]
print(label_list)
print('')

fname = 'test.csv'
keys = label_list[0].keys()
if not os.path.exists(fname):
    with open(fname, 'w') as csvfile:
        w = csv.DictWriter(csvfile, keys)
        w.writeheader()
        w.writerows(label_list)
else:
    with open(fname, 'a') as csvfile:
        w = csv.DictWriter(csvfile, keys)
        w.writerows(label_list)




#
