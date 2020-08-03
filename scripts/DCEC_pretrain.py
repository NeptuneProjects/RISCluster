import os
import sys
sys.path.insert(0, '../RISCluster/')

import numpy as np
import torch

import importlib as imp
import utils
imp.reload(utils)
import production
imp.reload(production)

# Universal Parameters ========================================================
mode = 'pretrain'
fname_dataset = '../../../Data/DetectionData.h5'
savepath = '../../../Outputs/'
# Pre-Training Routine ========================================================
if mode == 'pretrain':
    savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
    parameters = dict(
        fname_dataset = fname_dataset,
        device = utils.set_device(),
        M = 2400,
        n_epochs = 600,
        savepath = savepath_exp,
        serial = serial_exp,
        show = False,
        send_message = True,
        mode = mode,
        early_stopping = True,
        patience = 10
    )
    # hyperparameters = dict(
    #     batch_size = [256, 512],
    #     lr = [0.0001, 0.001]
    # )
    hyperparameters = dict(
        batch_size = [128, 256, 512, 1024],
        lr = [0.00001, 0.0001, 0.001]
    )
    production.DCEC_pretrain(parameters, hyperparameters)
