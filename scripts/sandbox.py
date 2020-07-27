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

M = int(1000)
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

fname = '../../../Outputs/Models/AEC/Run20200719T211701/AEC_Params_20200719T211701.pt'
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA device available, using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA device not available, using CPU.')
encoder, decoder, autoencoder = cluster.load_autoencoder(fname, device)


# def train(
#
# )


N_CLUSTERS = 11
features = autoencoder.encoder(X_train)
N_FEATURES = features.size(1)
ALPHA = 1.0

kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20)
labels = kmeans.fit_predict(features.detach().numpy())
labels_last = np.copy(labels)

decmodel = cluster.DEC(N_CLUSTERS, N_FEATURES, encoder, ALPHA)
optimizer = optim.Adam(decmodel.parameters(), lr=LR)
# Training function here:





decmodel.train()

predicted = kmeans.fit_predict(features.detach().numpy())
predicted.shape

#

























# End of script.
