#!/usr/bin/env python3

"""William Jenkins
Scripps Institution of Oceanography, UC San Diego
wjenkins [at] ucsd [dot] edu
May 2021

Contains high-level functions and routines for implementing the DEC
workflow.
"""

from itertools import product
import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from RISCluster import models, utils
from RISCluster.networks import AEC, DEC, init_weights


def load_data(config):

    if config.model == 'AEC' or config.model == 'DEC':

        dataset = utils.SeismicDataset(config.fname_dataset, config.datafiletype)

        if config.mode == 'train':
            index_tra, index_val = config.load_TraVal_index()
            config.index_tra = index_tra
            config.index_val = index_val
            tra_dataset = Subset(dataset, index_tra)
            if config.model == 'AEC':
                val_dataset = Subset(dataset, index_val)
            else:
                val_dataset = np.array([])
            del dataset

            if config.loadmode == 'ram':
                print("Loading Training Data to Memory:")
                tra_dataset = utils.dataset_to_RAM(tra_dataset)
                if config.model == 'AEC' and val_dataset is not None:
                    print("Loading Validation Data to Memory:")
                    val_dataset = utils.dataset_to_RAM(val_dataset)

            return tra_dataset, val_dataset

        elif config.mode == 'predict':
            return dataset

    elif config.model == 'GMM':
        fname = os.path.abspath(os.path.join(config.saved_weights, os.pardir))
        fname = os.path.join(fname, 'Prediction', 'Z_AEC.npy')
        dataset = np.load(fname)
        return dataset


def predict(config):
    dataset = load_data(config)
    print(f'Dataset has {len(dataset)} samples.')
    batch_size = 4096

    print('-'*100)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config.workers,
        pin_memory=True
    )

    if config.model == 'AEC':

        model = AEC().to(config.device)

        metrics = [nn.MSELoss(reduction='mean')]


    elif config.model == 'DEC':

        metrics = [
            nn.MSELoss(reduction='mean'),
            nn.CosineSimilarity(),
        ]

        model = DEC(config.n_clusters).to(config.device)

    models.model_prediction(
        config,
        model,
        dataloader,
        metrics,
    )

    return model


def train(config):
    tra_dataset, val_dataset = load_data(config)
    print(f'Dataset has {len(tra_dataset)+len(val_dataset)} samples.')
    print(f'  Training subset has {len(tra_dataset)} samples.')
    if config.model == 'AEC':
        print(f'  Validation subset has {len(val_dataset)} samples.')
    run_count = 1
    hp_keys = [k for k in config.hp.keys()]
    hp_vals = [v for v in config.hp.values()]
    hpkwargs = dict()

    for hp in product(*hp_vals):

        print('-'*100)
        print(f'Hyperparemeter Tuning Run {run_count}/{config.runs}')


        for i, k in enumerate(hp_keys):
            hpkwargs[k] = hp[i]

        batch_size, lr = hp[0], hp[1]

        tra_loader = DataLoader(
            tra_dataset,
            batch_size=batch_size,
            num_workers=config.workers,
            pin_memory=True
        )

        if config.model == "AEC":

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=config.workers,
                pin_memory=True
            )
            dataloaders = [tra_loader, val_loader]

            print(f"batch_size = {batch_size} | lr = {lr}")

            model = AEC().to(config.device)
            model.apply(init_weights)

            metrics = [nn.MSELoss(reduction='mean')]

        elif config.model == "DEC":
            print(str().join([f" {k} = {v} |" for k, v in hpkwargs.items()])[:-1])

            dataloaders = [tra_loader]

            model = DEC(n_clusters=int(hp[2])).to(config.device)

            metrics = [
                nn.MSELoss(reduction='mean'),
                nn.KLDivLoss(reduction='sum')
            ]

        optimizer = optim.Adam(model.parameters(), lr=lr)

        config.init_output_env(**hpkwargs)

        models.model_training(
            config,
            model,
            dataloaders,
            metrics,
            optimizer,
            **hpkwargs
        )
        run_count += 1

    return model


def gmm_fit(config):
    dataset = load_data(config)
    print(f'Dataset has {len(dataset)} samples.')

    run_count = 1

    for n_clusters in config.hp['n_clusters']:
        print('-'*100)
        print(f'GMM Run {run_count}/{config.runs}: n_clusters={n_clusters}')
        print('-' * 25)
        config.init_output_env(n_clusters=n_clusters)
        models.gmm_fit(config, dataset, n_clusters)
        run_count += 1
    return
