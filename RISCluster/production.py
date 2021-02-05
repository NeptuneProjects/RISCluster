#!/usr/bin/env python3

import argparse
import configparser
from datetime import datetime
from itertools import product
import os
import time

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from RISCluster import models, utils
from RISCluster.networks import AEC, DEC, init_weights


def DEC_pretrain(parameters, hyperparameters):
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
    workers = parameters['workers']
    # ==== Checks =============================================================
    if not os.path.exists(fname_dataset):
        raise ValueError(f'Dataset file not found: {fname_dataset}')
    if not os.path.exists(indexpath):
        raise ValueError(f'Index file not found: {indexpath}')
    # ==== Load Data ==========================================================
    index_tra, index_val = utils.load_TraVal_index(fname_dataset, indexpath)
    dataset = utils.H5SeismicDataset(
        fname_dataset,
        transform = transforms.Compose(
            [utils.SpecgramShaper(), utils.SpecgramToTensor()]
        )
    )
    tra_dataset = Subset(dataset, index_tra)
    val_dataset = Subset(dataset, index_val)
    print(f'Dataset has {len(dataset)} samples.')
    print(f'Training subset has {len(tra_dataset)} samples.')
    print(f'Validation subset has {len(val_dataset)} samples.')
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
                # ==== Instantiate Model, Optimizer, & Loss Functions =========
                model = AEC()
                # if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
                    # print(f'{torch.cuda.device_count()} GPUs in use.')
                    # model = nn.DataParallel(model)
                model.to(device)
                model.apply(init_weights)

                criterion_mse = nn.MSELoss(reduction='mean')
                criteria = [criterion_mse]

                optimizer = optim.Adam(model.parameters(), lr=lr)

                tra_loader = DataLoader(
                    tra_dataset,
                    batch_size=batch_size,
                    num_workers=workers
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    num_workers=workers
                )
                dataloaders = [tra_loader, val_loader]
                # ==== Pre-train DEC by training the autoencoder: =============
                _ = models.pretrain(
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
                raise
            finally:
                torch.cuda.empty_cache()

        if send_message:
            toc = datetime.now()
            msgsubj = 'DEC Pre-training & Tuning Status Update'
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
    msgcontent = f'DEC pre-training & tuning completed at {toc}.' + \
                 f'\nTime Elapsed = {toc-tic}.'
    print(msgcontent)
    if send_message:
        msgsubj = 'DEC Pre-training & Tuning Complete'
        utils.notify(msgsubj, msgcontent)

    print('To view results in Tensorboard, run the following command:')
    print(f'cd {savepath_exp} && tensorboard --logdir=.')
    print('==============================================================')


def DEC_train(parameters, hyperparameters):
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
    workers = parameters['workers']
    # ==== Checks =============================================================
    if not os.path.exists(saved_weights):
        raise ValueError(f'Saved weights file not found: {saved_weights}')
    if not os.path.exists(fname_dataset):
        raise ValueError(f'Dataset file not found: {fname_dataset}')
    if not os.path.exists(indexpath):
        raise ValueError(f'Index file not found: {indexpath}')
    # ==== Load Data ==========================================================
    index_tra, _ = utils.load_TraVal_index(fname_dataset, indexpath)
    dataset = utils.H5SeismicDataset(
        fname_dataset,
        transform = transforms.Compose(
            [utils.SpecgramShaper(), utils.SpecgramToTensor()]
        )
    )
    tra_dataset = Subset(dataset, index_tra)
    print(f'Dataset has {len(dataset)} samples.')
    print(f'Training subset has {len(tra_dataset)} samples.')
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
        model = DEC(n_clusters).to(device)

        criterion_mse = nn.MSELoss(reduction='mean')
        criterion_kld = nn.KLDivLoss(reduction='sum')
        criteria = [criterion_mse, criterion_kld]

        optimizer = optim.Adam(model.parameters(), lr=lr)

        dataloader = DataLoader(
            tra_dataset,
            batch_size=batch_size,
            num_workers=workers
        )
        # ==== Train DEC parameters: =========================================
        models.train(
            model,
            dataloader,
            criteria,
            optimizer,
            n_clusters,
            batch_size,
            lr,
            gamma,
            tol,
            index_tra,
            parameters
        )
        if send_message:
            toc = datetime.now()
            msgsubj = 'DEC Training & Tuning Status Update'
            msgcontent = f'Tuning run {tuning_count}/{tuning_runs} complete.'+\
                         f'\nTime Elapsed = {toc-tic}'
            utils.notify(msgsubj, msgcontent)
        tuning_count += 1

    print('--------------------------------------------------------------')
    toc = datetime.now()
    msgcontent = f'DEC training & tuning completed at {toc}.' + \
                 f'\nTime Elapsed = {toc-tic}.'
    print(msgcontent)
    if send_message:
        msgsubj = 'DEC Training & Tuning Complete'
        utils.notify(msgsubj, msgcontent)
    print('To view results in Tensorboard, run the following command:')
    print(f'cd {savepath_exp} && tensorboard --logdir=.')
    print('==============================================================')


def DEC_predict(parameters):
    print('==============================================================')
    print('Executing Prediction Mode')
    tic = datetime.now()
    # ==== Unpack Parameters ==================================================
    fname_dataset = parameters['fname_dataset']
    device = parameters['device']
    saved_weights = parameters['saved_weights']
    n_clusters = parameters['n_clusters']
    send_message = parameters['send_message']
    transform = parameters['transform']
    workers = parameters['workers']
    # ==== Checks =============================================================
    if not os.path.exists(saved_weights):
        raise ValueError(f'Saved weights file not found: {saved_weights}')
    if not os.path.exists(fname_dataset):
        raise ValueError(f'Dataset file not found: {fname_dataset}')
    # ==== Run Model ==========================================================
    dataset = utils.H5SeismicDataset(
        fname_dataset,
        transform = transforms.Compose(
            [utils.SpecgramShaper(), utils.SpecgramToTensor()]
        )
    )
    print(f'Dataset has {len(dataset)} samples.')
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=workers
    )
    model = DEC(n_clusters).to(device)
    models.predict(model, dataloader, parameters)
    print('--------------------------------------------------------------')
    toc = datetime.now()
    msgcontent = f'DEC outputs saved.\nTime Elapsed = {toc-tic}.'
    print(msgcontent)
    if send_message:
        msgsubj = 'DEC Outputs Saved'
        utils.notify(msgsubj, msgcontent)
    print('==============================================================')


def runDEC():
    print(__name__)
    """This command line function is the primary script that performs
    pre-training and training of the deep embedded clustering workflows.

    Parameters
    ----------
    init_file : str
        Path to the configuration file

    cuda_device : str
        CUDA device number (optional)
    """
    parser = argparse.ArgumentParser(
        description="Script pretrains or trains DEC model."
    )
    parser.add_argument('init_file', help="Enter path to init file.")
    parser.add_argument('--cuda_device', help="Select CUDA device.")
    args = parser.parse_args()
    init_file = args.init_file
    config = configparser.ConfigParser()
    config.read(init_file)

    matplotlib.use('Agg')

    if args.cuda_device is not None:
        device = utils.set_device(args.cuda_device)
    else:
        device = utils.set_device()
    # =========================================================================
    # Universal Parameters
    # =========================================================================
    fname_dataset = config['UNIVERSAL']['fname_dataset']
    savepath = config['UNIVERSAL']['savepath']
    indexpath = config['UNIVERSAL']['indexpath']
    mode = config['PARAMETERS']['mode']
    if mode != 'predict':
        savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
    if config['PARAMETERS'].getboolean('tb'):
        tbport = int(config['PARAMETERS']['tbport'])
        tbpid = utils.start_tensorboard(savepath_exp, tbport)
    else:
        tbpid = None
    # =========================================================================
    # Pre-Training Routine
    # =========================================================================
    if mode == 'pretrain':
        parameters = dict(
            fname_dataset=fname_dataset,
            device=device,
            indexpath=indexpath,
            n_epochs=int(config['PARAMETERS']['n_epochs']),
            savepath=savepath_exp,
            serial=serial_exp,
            show=config['PARAMETERS'].getboolean('show'),
            send_message=config['PARAMETERS'].getboolean('send_message'),
            mode=mode,
            early_stopping=config['PARAMETERS'].getboolean('early_stopping'),
            patience=int(config['PARAMETERS']['patience']),
            transform=config['PARAMETERS']['transform'],
            km_metrics=config['PARAMETERS'].getboolean('km_metrics'),
            klist=config['PARAMETERS']['klist'],
            img_index=config['PARAMETERS']['img_index'],
            tbpid=tbpid,
            workers=int(config['PARAMETERS']['workers'])
        )
        batch_size = config['HYPERPARAMETERS']['batch_size']
        lr = config['HYPERPARAMETERS']['lr']
        hyperparameters = dict(
            batch_size=[int(i) for i in batch_size.split(', ')],
            lr=[float(i) for i in lr.split(', ')]
        )
        utils.save_exp_config(
            savepath_exp,
            serial_exp,
            init_file,
            parameters,
            hyperparameters
        )
        DEC_pretrain(parameters, hyperparameters)
    # =========================================================================
    # Training Routine
    # =========================================================================
    if mode == 'train':
        parameters = dict(
            fname_dataset=fname_dataset,
            device=device,
            indexpath=indexpath,
            n_epochs=int(config['PARAMETERS']['n_epochs']),
            update_interval=int(config['PARAMETERS']['update_interval']),
            savepath=savepath_exp,
            serial=serial_exp,
            show=config['PARAMETERS'].getboolean('show'),
            send_message=config['PARAMETERS'].getboolean('send_message'),
            mode=mode,
            saved_weights=config['PARAMETERS']['saved_weights'],
            transform=config['PARAMETERS']['transform'],
            tbpid=tbpid,
            workers=int(config['PARAMETERS']['workers']),
            init=config['PARAMETERS']['init']
        )
        n_clusters = config['HYPERPARAMETERS']['n_clusters']
        batch_size = config['HYPERPARAMETERS']['batch_size']
        lr = config['HYPERPARAMETERS']['lr']
        gamma = config['HYPERPARAMETERS']['gamma']
        tol = config['HYPERPARAMETERS']['tol']
        hyperparameters = dict(
            n_clusters=[int(i) for i in n_clusters.split(', ')],
            batch_size=[int(i) for i in batch_size.split(', ')],
            lr=[float(i) for i in lr.split(', ')],
            gamma=[float(i) for i in gamma.split(', ')],
            tol=[float(i) for i in tol.split(', ')]
        )
        utils.save_exp_config(
            savepath_exp,
            serial_exp,
            init_file,
            parameters,
            hyperparameters
        )
        DEC_train(parameters, hyperparameters)
    # =========================================================================
    # Prediction Routine
    # =========================================================================
    if mode == 'predict':
        saved_weights=config['PARAMETERS']['saved_weights']
        parameters = dict(
            fname_dataset=fname_dataset,
            device=device,
            saved_weights=saved_weights,
            n_clusters=int(utils.parse_nclusters(saved_weights)),
            send_message=config['PARAMETERS'].getboolean('send_message'),
            transform=config['PARAMETERS']['transform'],
            workers=int(config['PARAMETERS']['workers'])
        )
        DEC_predict(parameters)

def main():
    print("I am Main.")

if __name__ == '__main__':
    main()
