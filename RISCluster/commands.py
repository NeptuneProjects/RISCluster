#!/usr/bin/env python3

"""Command line tools for RISCluster.

William Jenkins, wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography, UC San Diego
January 2021
"""
import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import configparser
import json
import os

import h5py
import numpy as np
import matplotlib
from tqdm import tqdm

from RISCluster import production, utils


def runDEC():
    """This command line function is the primary script that performs
    pre-training and training of the deep embedded clustering workflows.

    Parameters
    ----------
    init_file : str
        Path to the configuration file

    cuda_device : str
        CUDA device number (optional)
    """
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(
        description="Script pretrains or trains DEC model."
    )
    parser.add_argument('init_file', help="Enter path to init file.")
    parser.add_argument('--cuda_device', help="Select CUDA device.")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(init_file)

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
        production.DCM_pretrain(parameters, hyperparameters)
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
        production.DCM_train(parameters, hyperparameters)
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
        production.DCM_predict(parameters)


def query_H5size():
    """Command line function that prints the dimensions of the specified HDF
    database.

    Parameters
    ----------
    path : str
        Path to the H5 database.
    """
    parser = argparse.ArgumentParser(description='Enter path to .h5 file.')
    parser.add_argument(
        'path',
        help='Enter path to database; must be .h5/.hd5 file.'
    )
    args = parser.parse_args()
    path = args.path
    m, n, o = query_dbSize(path)
    print(f" >> h5 dataset contains {m} samples with dimensions [{n},{o}]. <<")
    pass


def ExtractH5Dataset():

    def _copy_attributes(in_object, out_object):
        """Copy attributes between 2 HDF5 objects.

        Parameters
        ----------
        in_object : object
            Object to be copied

        out_ojbect : object
            Object to which in_object attributes are copied.
        """
        for key, value in in_object.attrs.items():
            out_object.attrs[key] = value


    def _find_indeces(index, source, stations):
        """Returns index of sample if it matches station input.

        Parameters
        ----------
        index : int
            Sample index to be queried.

        source : str
            Path to H5 database.

        stations : list
            List of stations to match.

        Returns
        -------
        index : int
            Sample index if match; nan otherwise.
        """
        with h5py.File(source, 'r') as f:
            metadata = json.loads(f['/4.0/Catalogue'][index])
        if metadata["Station"] in stations:
            return index
        else:
            return np.nan

    parser = argparse.ArgumentParser(
        description="Creates new dataset from existing dataset."
    )
    parser.add_argument("source", help="Enter path to source dataset.")
    parser.add_argument("dest", help="Enter path to destination dataset.")
    parser.add_argument("--include", help="Enter stations to include.")
    parser.add_argument("--exclude", help="Enter stations to exclude.")
    parser.add_argument("--after", help="Include after YYYYMMDDTHHMMSS.")
    parser.add_argument("--before", help="Include before YYYYMMDDTHHMMSS.")
    args = parser.parse_args()

    if args.include is None and args.exclude is None:
        raise Exception("Must specify stations to include or exclude.")
    source = args.source
    dest = args.dest
    include = args.include
    exclude = args.exclude
    after = args.after
    before = args.before

    if not os.path.exists(source):
        raise ValueError(f"Source file not found: {source}")

    with h5py.File(source, 'r') as rf:
        M = len(rf['/4.0/Trace'])
    index = np.arange(1, M)

    if include is not None:
        include = json.loads(include)
    else:
        include = None
    if exclude is not None:
        exclude = json.loads(exclude)
    else:
        exclude = None
    if after is not None:
        after = after
    else:
        after = None
    if before is not None:
        before = before
    else:
        before = None

    if include is not None and exclude is not None:
        removals = [processing.get_station(i) for i in exclude]
        stations = [processing.get_station(i) for i in include]
        stations = [i for i in stations if i not in removals]
        print(f"Searching {stations}")
    elif include is not None:
        stations = [processing.get_station(i) for i in include]
        print(f"Searching {stations}")
    elif exclude is not None:
        removals = [processing.get_station(i) for i in exclude]
        stations = [processing.get_station(i) for i in range(34)]
        stations = [i for i in stations if i not in removals]
        print(f"Searching {stations}")
    else:
        stations = [processing.get_station(i) for i in range(34)]
        print(f"Searching {stations}")

    A = [{
            "index": index[i],
            "source": source,
            "stations": stations
        } for i in range(len(index))]

    index_keep = np.zeros(len(index))
    with ProcessPoolExecutor(max_workers=14) as exec:
        print("Finding detections that meet filter criteria...")
        futures = [exec.submit(_find_indeces, **a) for a in A]
        kwargs1 = {
            "total": int(len(index)),
            "bar_format": '{l_bar}{bar:20}{r_bar}{bar:-20b}',
            "leave": True
        }
        for i, future in enumerate(tqdm(as_completed(futures), **kwargs1)):
            index_keep[i] = future.result()
    index_keep = np.sort(index_keep[~np.isnan(index_keep)]).astype(int)

    with h5py.File(source, 'r') as fs, h5py.File(dest, 'w') as fd:
        M = len(index_keep)
        dset_names = ['Catalogue', 'Trace', 'Spectrogram', 'Scalogram']
        # dset_names = ['Catalogue']
        for dset_name in dset_names:
            group_path = '/4.0'
            dset = fs[f"{group_path}/{dset_name}"]
            dset_shape = dset.shape[1:]
            dset_shape = (M,) + dset_shape
            group_id = fd.require_group(group_path)
            dset_id = group_id.create_dataset(dset_name, dset_shape, dtype=dset.dtype, chunks=None)
            _copy_attributes(dset, dset_id)
            kwargs2 = {
                "desc": dset_name,
                "bar_format": '{l_bar}{bar:20}{r_bar}{bar:-20b}'
            }
            for i in tqdm(range(len(index_keep)), **kwargs2):
                dset_id[i] = dset[index_keep[i]]
