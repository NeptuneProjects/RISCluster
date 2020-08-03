from datetime import datetime
import sys
sys.path.insert(0, '../RISCluster/')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import importlib as imp
import model
imp.reload(model)
from model import train_DCEC

import networks
imp.reload(networks)
from  networks import Encoder, Decoder, AEC, DCEC

import utils
imp.reload(utils)
from utils import init_dcec_output_env, load_data, notify, save_history, set_loading_index
# =============================================================================
# Initialize environment and load data.
# =============================================================================
print('==============================================================')
fname_dataset = '../../../Data/DetectionData.h5'
savepath_run, run_serial = init_dcec_output_env()

M = int(2000)
M_train = int(0.8 * M)
M_val = int(0.2 * M)
M_test = M
LR = 0.0001     # Learning rate
N_EPOCHS = 100  # Number of epochs
BATCH_SZ = 256  # Batch size

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

# Instantiate DCEC model:
model = DCEC(n_clusters=11).to(device)

# Define Loss Metrics:
criterion1 = nn.MSELoss(reduction='mean')
criterion2 = nn.KLDivLoss(reduction='sum')
criteria = [criterion1, criterion2]

# Define Optimizer:
optimizer = optim.Adam(model.parameters(), lr=LR)

params = {}
params['batch_size'] = BATCH_SZ
params['update_interval'] = 5
params['device'] = device
params['gamma'] = 0.11
params['dataset_size'] = M_train
params['tol'] = 0.001
params['run_serial'] = run_serial
params['savepath'] = savepath_run
params['send_message'] = False
n_epochs = N_EPOCHS
trained = train_DCEC(model, train_loader, criteria, optimizer, N_EPOCHS, params)

# End of script.
