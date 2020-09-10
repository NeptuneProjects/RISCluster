import argparse
import configparser
import os
import sys
sys.path.insert(0, '../RISCluster/')

import production
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script pretrains or trains DCM."
    )
    parser.add_argument('init_file', help="Enter path to init file.")
    parser.add_argument('--cuda_device', help="Select CUDA device.")
    args = parser.parse_args()
    init_file = args.init_file
    config = configparser.ConfigParser()
    # init_file = '../init_predict.ini'
    config.read(init_file)

    if args.cuda_device is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # =========================================================================
    # Universal Parameters
    # =========================================================================
    mode = config['UNIVERSAL']['mode']
    fname_dataset = config['UNIVERSAL']['fname_dataset']
    savepath = config['UNIVERSAL']['savepath']
    indexpath = config['UNIVERSAL']['indexpath']
    # =========================================================================
    # Pre-Training Routine
    # =========================================================================
    if mode == 'pretrain':
        savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
        parameters = dict(
            fname_dataset=fname_dataset,
            device=utils.set_device(),
            indexpath=indexpath,
            n_epochs=int(config['PARAMETERS']['n_epochs']),
            savepath=savepath_exp,
            serial=serial_exp,
            show=config['PARAMETERS'].getboolean('show'),
            send_message=config['PARAMETERS'].getboolean('send_message'),
            mode=mode,
            early_stopping=config['PARAMETERS'].getboolean('early_stopping'),
            patience=int(config['PARAMETERS']['patience']),
            transform=config['PARAMETERS']['transform']
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
        production.DCM_pretrain(parameters, hyperparameters)
    # =========================================================================
    # Training Routine
    # =========================================================================
    if mode == 'train':
        savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
        parameters = dict(
            fname_dataset=fname_dataset,
            device=utils.set_device(),
            indexpath=indexpath,
            n_epochs=int(config['PARAMETERS']['n_epochs']),
            update_interval=int(config['PARAMETERS']['update_interval']),
            savepath=savepath_exp,
            serial=serial_exp,
            show=config['PARAMETERS'].getboolean('show'),
            send_message=config['PARAMETERS'].getboolean('send_message'),
            mode=mode,
            saved_weights=config['PARAMETERS']['saved_weights'],
            transform=config['PARAMETERS']['transform']
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
        production.DCM_train(parameters, hyperparameters)
    # =========================================================================
    # Prediction Routine
    # =========================================================================
    if mode == 'predict':
        savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
        M = config['PARAMETERS']['m'] # Select integer or 'all'
        if M == 'all':
            pass
        else:
            M = int(M)
        parameters = dict(
            fname_dataset=fname_dataset,
            device=utils.set_device(),
            M = M,
            indexpath=indexpath,
            exclude=config['PARAMETERS'].getboolean('exclude'),
            batch_size=int(config['PARAMETERS']['batch_size']),
            n_clusters=int(config['PARAMETERS']['n_clusters']),
            savepath=savepath_exp,
            serial=serial_exp,
            show=config['PARAMETERS'].getboolean('show'),
            send_message=config['PARAMETERS'].getboolean('send_message'),
            mode=mode,
            saved_weights=config['PARAMETERS']['saved_weights'],
            max_workers=int(config['PARAMETERS']['max_workers']),
            loaded=False,
            transform=config['PARAMETERS']['transform']
        )
        utils.save_exp_config(
            savepath_exp,
            serial_exp,
            init_file,
            parameters,
            None
        )
        production.DCM_predict(parameters)
