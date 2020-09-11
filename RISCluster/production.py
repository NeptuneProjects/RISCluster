from datetime import datetime
from itertools import product
import os
import sys
sys.path.insert(0, '../RISCluster/')
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import importlib as imp
import models
from networks import AEC, DCM, init_weights
import utils

def DCM_pretrain(parameters, hyperparameters):
    print('==============================================================')
    print('Executing Pre-training Mode')
    tic = datetime.now()
    # ==== Unpack Parameters ==================================================
    fname_dataset = parameters['fname_dataset']
    # M = parameters['M']
    show = parameters['show']
    send_message = parameters['send_message']
    mode = parameters['mode']
    device = parameters['device']
    indexpath = parameters['indexpath']
    savepath_exp = parameters['savepath']
    transform = parameters['transform']
    # ==== Checks =============================================================
    if not os.path.exists(fname_dataset):
        raise ValueError(f'Dataset file not found: {fname_dataset}')
    if not os.path.exists(indexpath):
        raise ValueError(f'Index file not found: {indexpath}')
    # ==== Load Data ==========================================================
    index_tra, index_val = utils.load_TraVal_index(fname_dataset, indexpath)
    M_tra = len(index_tra)
    M_val = len(index_val)

    tra_dataset = utils.batch_load(
        fname_dataset,
        index_tra,
        send_message=send_message,
        transform=transform
    )
    val_dataset = utils.batch_load(
        fname_dataset,
        index_val,
        send_message=send_message,
        transform=transform
    )
    # ==== Commence Pre-training ==============================================
    hyperparam_values = [v for v in hyperparameters.values()]
    tuning_runs = utils.calc_tuning_runs(hyperparameters)
    tuning_count = 1
    for batch_size, lr in product(*hyperparam_values):
        completed = False
        oom_attempt = 0
        print('--------------------------------------------------------------')
        print(f'Hyperparemeter Tuning Run {tuning_count}/{tuning_runs}')
        print(f'Batch Size = {batch_size}, LR = {lr}')
        print('To view results in Tensorboard, run the following command:')
        print(f'cd {savepath_exp} && tensorboard --logdir=.')
        queued_flag = False
        q_tic = datetime.now()
        while not completed:
            try:
                # ==== Instantiate Model, Optimizer, & Loss Functions =================
                model = AEC()
                # if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
                    # print(f'{torch.cuda.device_count()} GPUs in use.')
                    # model = nn.DataParallel(model)
                model.to(device)
                model.apply(init_weights)

                criterion_mse = nn.MSELoss(reduction='mean')
                criterion_mae = nn.L1Loss(reduction='mean')
                criteria = [criterion_mse, criterion_mae]

                optimizer = optim.Adam(model.parameters(), lr=lr)

                tra_loader = DataLoader(tra_dataset, batch_size=batch_size)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                dataloaders = [tra_loader, val_loader]
                # ==== Pre-train DCM parameters by training the autoencoder: =========
                models.pretrain_DCM(
                    model,
                    dataloaders,
                    criteria,
                    optimizer,
                    batch_size,
                    lr,
                    parameters
                )
                completed = True
            except RuntimeError as e:
                if ('CUDA' and 'out of memory') in str(e):
                    queued_flag = True
                    oom_attempt += 1
                    torch.cuda.empty_cache()
                    queued = str(datetime.now() - q_tic)
                    print(
                        f'| WARNING: Out of memory, queued for: {queued[:-7]}',
                        end='\r'
                    )
                    time.sleep(1)
                else:
                    raise e
            except KeyboardInterrupt:
                print('Re-attempt terminated by user, ending program.')
                torch.cuda.empty_cache()
            finally:
                torch.cuda.empty_cache()

        if send_message:
            toc = datetime.now()
            msgsubj = 'DCM Pre-training & Tuning Status Update'
            msgcontent = f'Tuning run {tuning_count}/{tuning_runs} complete.'+\
                         f'\nTime Elapsed = {toc-tic}'
            if queued_flag:
                msgcontent = msgcontent + '\nBefore this tuning run, your' + \
                    'program encountered an out-of-memory error on CUDA ' + \
                    f'device and was queued for {queued[:-7]}.'
                queued_flag = False
            utils.notify(msgsubj, msgcontent)
        tuning_count += 1

    print('--------------------------------------------------------------')
    toc = datetime.now()
    msgcontent = f'DCM pre-training & tuning completed at {toc}.' + \
                 f'\nTime Elapsed = {toc-tic}.'
    print(msgcontent)
    if send_message:
        msgsubj = 'DCM Pre-training & Tuning Complete'
        utils.notify(msgsubj, msgcontent)

    print('To view results in Tensorboard, run the following command:')
    print(f'cd {savepath_exp} && tensorboard --logdir=.')
    print('==============================================================')

def DCM_train(parameters, hyperparameters):
    print('==============================================================')
    print('Executing Training Mode')
    tic = datetime.now()
    # ==== Unpack Parameters ==================================================
    fname_dataset = parameters['fname_dataset']
    show = parameters['show']
    send_message = parameters['send_message']
    mode = parameters['mode']
    device = parameters['device']
    savepath_exp = parameters['savepath']
    indexpath = parameters['indexpath']
    saved_weights = parameters['saved_weights']
    transform = parameters['transform']
    # ==== Checks =============================================================
    if not os.path.exists(saved_weights):
        raise ValueError(f'Saved weights file not found: {saved_weights}')
    if not os.path.exists(fname_dataset):
        raise ValueError(f'Dataset file not found: {fname_dataset}')
    if not os.path.exists(indexpath):
        raise ValueError(f'Index file not found: {indexpath}')
    # ==== Load Data ==========================================================
    index_tra, _ = utils.load_TraVal_index(fname_dataset, indexpath)
    M_tra = len(index_tra)
    tra_dataset = utils.batch_load(
        fname_dataset,
        index_tra,
        send_message=send_message,
        transform=transform
    )
    # ==== Commence Training ==================================================
    hyperparam_values = [v for v in hyperparameters.values()]
    tuning_runs = utils.calc_tuning_runs(hyperparameters)
    tuning_count = 1
    for n_clusters, batch_size, lr, gamma, tol in product(*hyperparam_values):
        print('--------------------------------------------------------------')
        print(f'Hyperparemeter Tuning Run {tuning_count}/{tuning_runs}')
        print(
            f'# Clusters = {n_clusters}, Batch Size = {batch_size}, LR = {lr}, '
            f'gamma = {gamma}, tol = {tol}'
        )
        print('To view results in Tensorboard, run the following command:')
        print(f'cd {savepath_exp} && tensorboard --logdir=.')
        # ==== Instantiate Model, Optimizer, & Loss Functions =================
        model = DCM(n_clusters).to(device)

        criterion_mse = nn.MSELoss(reduction='mean')
        criterion_kld = nn.KLDivLoss(reduction='sum')
        criteria = [criterion_mse, criterion_kld]

        optimizer = optim.Adam(model.parameters(), lr=lr)

        dataloader = DataLoader(tra_dataset, batch_size=batch_size)
        # ==== Train DCM parameters: =========================================
        models.train_DCM(
            model,
            dataloader,
            criteria,
            optimizer,
            n_clusters,
            batch_size,
            lr,
            gamma,
            tol,
            parameters
        )
        if send_message:
            toc = datetime.now()
            msgsubj = 'DCM Training & Tuning Status Update'
            msgcontent = f'Tuning run {tuning_count}/{tuning_runs} complete.'+\
                         f'\nTime Elapsed = {toc-tic}'
            utils.notify(msgsubj, msgcontent)
        tuning_count += 1

    print('--------------------------------------------------------------')
    toc = datetime.now()
    msgcontent = f'DCM training & tuning completed at {toc}.' + \
                 f'\nTime Elapsed = {toc-tic}.'
    print(msgcontent)
    if send_message:
        msgsubj = 'DCM Training & Tuning Complete'
        utils.notify(msgsubj, msgcontent)
    print('To view results in Tensorboard, run the following command:')
    print(f'cd {savepath_exp} && tensorboard --logdir=.')
    print('==============================================================')

def DCM_predict(parameters, index_tst=None, tst_dataset=None):
    print('==============================================================')
    print('Executing Prediction Mode')
    tic = datetime.now()
    # ==== Unpack Parameters ==================================================
    fname_dataset = parameters['fname_dataset']
    device = parameters['device']
    M = parameters['M']
    batch_size = parameters['batch_size']
    n_clusters = parameters['n_clusters']
    show = parameters['show']
    send_message = parameters['send_message']
    mode = parameters['mode']
    saved_weights = parameters['saved_weights']
    indexpath = parameters['indexpath']
    exclude = parameters['exclude']
    loaded = parameters['loaded']
    transform = parameters['transform']
    # ==== Checks =============================================================
    if not os.path.exists(saved_weights):
        raise ValueError(f'Saved weights file not found: {saved_weights}')
    if not os.path.exists(fname_dataset):
        raise ValueError(f'Dataset file not found: {fname_dataset}')
    if not os.path.exists(indexpath):
        raise ValueError(f'Index file not found: {indexpath}')
    # ==== Load Data ==========================================================
    if not loaded:
        if isinstance(M, str) and (M == 'all'):
            M = utils.set_M(fname_dataset, indexpath, exclude=exclude)
        index_tst = utils.set_Tst_index(
            M,
            fname_dataset,
            indexpath,
            exclude=exclude
        )
        tst_dataset = utils.batch_load(
            fname_dataset,
            index_tst,
            send_message,
            transform=transform
        )

    dataloader = DataLoader(tst_dataset, batch_size=batch_size)
    model = DCM(n_clusters).to(device)

    models.predict_DCM(model, dataloader, index_tst, parameters)
    print('--------------------------------------------------------------')
    toc = datetime.now()
    msgcontent = f'DCM outputs saved.\nTime Elapsed = {toc-tic}.'
    print(msgcontent)
    if send_message:
        msgsubj = 'DCM Outputs Saved'
        utils.notify(msgsubj, msgcontent)
    print('==============================================================')
