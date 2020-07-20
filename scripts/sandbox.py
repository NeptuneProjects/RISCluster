from datetime import datetime
import os
import sys
sys.path.insert(0, '../../RISClusterPT/')

import h5py
# from ignite.engine import Engine, Events
# from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError, Loss
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from RISCluster.processing import cluster
from RISCluster.processing.cluster import load_data, set_loading_index
from RISCluster.utils.utils import notify

import importlib
importlib.reload(cluster)

# =============================================================================
# Initialize environment and load data.
# =============================================================================
print('==============================================================')
fname_dataset = '../../../Data/DetectionData.h5'
savepath_run, _, _, run_serial = cluster.init_aec_output_env()

M = int(300000)
M_train = int(0.8 * M)
M_val = int(0.2 * M)
M_test = M
LR = 0.0001     # Learning rate
N_EPOCHS = 600  # Number of epochs
BATCH_SZ = 512  # Batch size

index_train, index_val, index_test = set_loading_index(
    M,
    fname_dataset,
    reserve=0.02
)

X_train, m, p, n, o, idx_smpl_train = load_data(
    fname_dataset,
    M_train,
    index_train,
    send_message=False
)

X_val, m, p, n, o, idx_smpl_val = load_data(
    fname_dataset,
    M_val,
    index_val,
    send_message=False
)

train_loader = DataLoader(X_train, batch_size=BATCH_SZ)
val_loader = DataLoader(X_val, batch_size=BATCH_SZ)

# =============================================================================
# Print examples of spectrograms
# =============================================================================
# insp_idx = sorted(np.random.randint(0,len(X_train),4))
# fixed_images = X_train[insp_idx,:,:,:].to(device)
#
# figtitle = 'Input Spectrograms'
# fig = cluster.view_specgram(
#     X_train,
#     insp_idx,
#     n,
#     o,
#     fname_dataset,
#     idx_smpl_train,
#     figtitle,
#     nrows=2,
#     ncols=2,
#     figsize=(12,9),
#     show=True
# )
# fname = savepath_fig + '01_InputSpecGrams_' + \
#         datetime.now().strftime('%Y%m%dT%H%M%S') + '.png'
# fig.savefig(fname)
# =============================================================================
# Pre-train DEC parameters by training the autoencoder:
# =============================================================================
autoencoder, pretraining_history, validation_history = cluster.pretrain(
    train_loader,
    val_loader,
    run_serial,
    epochs=N_EPOCHS,
    batch_size=BATCH_SZ,
    LR=LR,
    show=False,
    send_message=True,
    savepath = savepath_run
)

print('==============================================================')


# dict ={'mse': [3.34312431, 4, 76, 2, 5.]}
# len(dict['mse'])
#
# print(f'MSE is {dict["mse"]}')
# print('MSE is {}'.format(dict['mse']))
# Load the model:
# autoencoder = Autoencoder()
# autoencoder.load_state_dict(torch.load(fname))
# autoencoder.eval()
# encoder = autoencoder.encoder.state_dict()
# End of script.
