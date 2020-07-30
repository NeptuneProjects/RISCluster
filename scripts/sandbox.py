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

N_CLUSTERS = 11
features = autoencoder.encoder(X_train)
N_FEATURES = features.size(1)
ALPHA = 1.0
dcecmodel = cluster.DCEC(N_CLUSTERS, N_FEATURES, autoencoder, ALPHA)
optimizer = optim.Adam(decmodel.parameters(), lr=LR)



f, x_pred = dcecmodel(X_train)
print(f.size())
print(x_pred.size())

######### def train(
#
# )



kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20)
labels = kmeans.fit_predict(features.detach().numpy())
labels_last = np.copy(labels)

cluster_centers = torch.tensor(
    kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
).to(device)
with torch.no_grad():
    decmodel.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)

loss_mse = nn.MSELoss()
loss_kld = nn.KLDivLoss(reduction='sum')
delta_label = None


def training_step(engine, batch):
    dcecmodel.train()
    optimizer.zero_grad()
    x = batch.to(device)
    q, x_pred = dcecmodel(x)
    p = target_distribution(q)
    KLD = loss_kld(q, p)
    MSE = loss_mse(x_pred, x)
    loss = MSE + 0.11 * KLD
    loss.backward()
    optimizer.step()
    return loss.item(), KLD.item()

trainer = Engine(training_step)

def validation_step(engine, batch):
    dcecmodel.eval()
    with torch.no_grad():
        x = batch.to(device)
        q, x_pred = dcecmodel(x)
        p = target_distribution(q)
        return x_pred, x, q, p

evaluator = Engine(validation_step)

MeanSquaredError(device=device).attach(evaluator)


# Training function here:

# decmodel.train()
#

























# End of script.
