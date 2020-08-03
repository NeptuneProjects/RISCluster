from datetime import datetime
from itertools import product
import sys
sys.path.insert(0, '../RISCluster/')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import importlib as imp
import models
imp.reload(models)
from networks import AEC, DCEC, init_weights
import utils
imp.reload(utils)

def DCEC_pretrain(parameters, hyperparameters):
    print('==============================================================')
    tic = datetime.now()
    # ==== Unpack Parameters ==================================================
    fname_dataset = parameters['fname_dataset']
    M = parameters['M']
    show = parameters['show']
    send_message = parameters['send_message']
    mode = parameters['mode']
    device = parameters['device']
    # ==== Load Data ==========================================================
    M_tra = int(0.8 * M)
    M_val = int(0.2 * M)
    index_tra, index_val, _ = utils.set_loading_index(
        M,
        fname_dataset,
        reserve=0.02
    )
    X_tra, m, p, n, o, idx_smpl_tra = utils.load_data(
        fname_dataset,
        M_tra,
        index_tra,
        send_message
    )
    X_val, m, p, n, o, idx_smpl_val = utils.load_data(
        fname_dataset,
        M_val,
        index_val,
        send_message
    )
    # ==== Commence Pre-training ==============================================
    hyperparam_values = [v for v in hyperparameters.values()]
    tuning_runs = utils.calc_tuning_runs(hyperparameters)
    tuning_count = 1
    for batch_size, lr in product(*hyperparam_values):
        print('--------------------------------------------------------------')
        print(f'Hyperparemeter Tuning Run {tuning_count}/{tuning_runs}')
        print(f'Batch Size = {batch_size}, LR = {lr}')
        # ==== Instantiate Model, Optimizer, & Loss Functions =================
        model = AEC().to(device)
        model.apply(init_weights)

        criterion_mse = nn.MSELoss(reduction='mean')
        criterion_mae = nn.L1Loss(reduction='mean')
        criteria = [criterion_mse, criterion_mae]

        optimizer = optim.Adam(model.parameters(), lr=lr)

        tra_loader = DataLoader(X_tra, batch_size=batch_size)
        val_loader = DataLoader(X_val, batch_size=batch_size)
        dataloaders = [tra_loader, val_loader]
        # ==== Pre-train DCEC parameters by training the autoencoder: =========
        models.pretrain_DCEC(
            model,
            dataloaders,
            criteria,
            optimizer,
            batch_size,
            lr,
            parameters
        )
        tuning_count += 1

    print('--------------------------------------------------------------')
    toc = datetime.now()
    msgcontent = f'''DCEC pre-training & tuning completed at {toc}.
    Time Elapsed = {(toc-tic)}.'''
    print(msgcontent)
    if send_message:
        msgsubj = 'DCEC Pre-training & Tuning Complete'
        utils.notify(msgsubj, msgcontent)
    print('==============================================================')

def DCEC_train(parameters, hyperparameters):
    print('==============================================================')
    tic = datetime.now()
    # ==== Unpack Parameters ==================================================
    fname_dataset = parameters['fname_dataset']
    M = parameters['M']
    n_clusters = parameters['n_clusters']
    show = parameters['show']
    send_message = parameters['send_message']
    mode = parameters['mode']
    device = parameters['device']
    # ==== Load Data ==========================================================
    M_tra = int(0.8 * M)
    M_val = int(0.2 * M)
    index_tra, _, _ = utils.set_loading_index(
        M,
        fname_dataset,
        reserve=0.02
    )
    X_tra, m, p, n, o, idx_smpl_tra = utils.load_data(
        fname_dataset,
        M_tra,
        index_tra,
        send_message
    )
    # ==== Commence Training ==================================================
    hyperparam_values = [v for v in hyperparameters.values()]
    tuning_runs = utils.calc_tuning_runs(hyperparameters)
    tuning_count = 1
    for batch_size, lr, gamma, tol in product(*hyperparam_values):
        print('--------------------------------------------------------------')
        print(f'Hyperparemeter Tuning Run {tuning_count}/{tuning_runs}')
        print(
            f'Batch Size = {batch_size}, LR = {lr}, '
            f'gamma = {gamma}, tol = {tol}'
        )
        # ==== Instantiate Model, Optimizer, & Loss Functions =================
        model = DCEC(n_clusters).to(device)

        criterion_mse = nn.MSELoss(reduction='mean')
        criterion_kld = nn.KLDivLoss(reduction='sum')
        criteria = [criterion_mse, criterion_kld]

        optimizer = optim.Adam(model.parameters(), lr=lr)

        dataloader = DataLoader(X_tra, batch_size=batch_size)

        # ==== Train DCEC parameters: =========================================
        models.train_DCEC(
            model,
            dataloader,
            criteria,
            optimizer,
            batch_size,
            lr,
            gamma,
            tol,
            parameters
        )
        tuning_count += 1

    print('--------------------------------------------------------------')
    toc = datetime.now()
    msgcontent = f'''DCEC training & tuning completed at {toc}.
    Time Elapsed = {toc-tic}.'''
    print(msgcontent)
    if send_message:
        msgsubj = 'DCEC Training & Tuning Complete'
        utils.notify(msgsubj, msgcontent)
    print('==============================================================')

def DCEC_predict(parameters):
    print('==============================================================')
    fname_dataset = parameters['fname_dataset']
    device = parameters['device']
    M = parameters['M']
    batch_size = parameters['batch_size']
    n_clusters = parameters['n_clusters']
    show = parameters['show']
    send_message = parameters['send_message']
    mode = parameters['mode']
    saved_weights = parameters['saved_weights']

    _, _, index_tst = utils.set_loading_index(
        M,
        fname_dataset,
        reserve=0.02
    )
    X_tst, m, p, n, o, idx_smpl_tst = utils.load_data(
        fname_dataset,
        M,
        index_tst,
        send_message
    )

    dataloader = DataLoader(X_tst, batch_size=batch_size)
    model = DCEC(n_clusters).to(device)

    models.predict_DCEC(model, dataloader, idx_smpl_tst, parameters)
    print('==============================================================')
