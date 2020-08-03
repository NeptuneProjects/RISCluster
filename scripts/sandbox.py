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
from tqdm import tqdm

import importlib as imp

import networks
imp.reload(networks)
from  networks import AEC, DCEC

import production
imp.reload(production)

import utils
imp.reload(utils)

# =============================================================================
# Universal Parameters
# =============================================================================
mode = 'pretrain'
fname_dataset = '../../../Data/DetectionData.h5'
savepath = '../../../Outputs/'
loadpath = '/Users/williamjenkins/Research/Workflows/RIS_Clustering/Outputs/Models/AEC/Exp20200802T013941/Run_BatchSz=512_LR=0.0001/AEC_Params_20200802T061234.pt'
device = utils.set_device()

model = DCEC(n_clusters=14)
model.load_state_dict(torch.load(loadpath, map_location=device), strict=False)

# autoencoder = AEC()
# autoencoder.load_state_dict(torch.load(loadpath, map_location=device))
# autoencoder.eval()
# # print(list(autoencoder.named_parameters()))
#
# model = DCEC(n_clusters = 14)
# print(autoencoder)
# print(model)
#
# def copy_params(module_src, module_dest):
#     params_src = module_src.named_parameters()
#     params_dest = module_dest.named_parameters()
#     dict_dest = dict(params_dest)
#     for name, param in params_src:
#         if name in dict_dest:
#             print(name)
#             dict_dest[name].data.copy_(param.data)
#
# copy_params(autoencoder.encoder, model.encoder)
# copy_params(autoencoder.decoder, model.decoder)
# End of script.
