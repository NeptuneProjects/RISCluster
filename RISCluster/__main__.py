#!/usr/bin/env python3

import argparse
import configparser
import multiprocessing as mp

import matplotlib

from RISCluster import production, utils

def main():
    print("This is the MAIN function.")
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
        production.DEC_pretrain(parameters, hyperparameters)
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
        production.DEC_train(parameters, hyperparameters)
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
        production.DEC_predict(parameters)


if __name__ == '__main__':
    print(f"My name is {__name__}!!!")
    mp.set_start_method('spawn')
    main()
