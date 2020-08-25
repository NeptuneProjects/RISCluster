import argparse
import configparser
from datetime import datetime
import os
import re
import sys
sys.path.insert(0, '../RISCluster/')
import time

import production
import utils

loadpath = '../../../Outputs/Models/DCM'
savepath = '../../ConfigFiles'
overwrite = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Enter model mode and configuration file."
    )
    parser.add_argument('init_path', help="Enter path to folder containing init files.")
    args = parser.parse_args()
    init_path = args.init_path

    # init_path = '../../ConfigFiles/BatchEval0824T2044'

    # =========================================================================
    # Universal Parameters
    # =========================================================================
    mode = 'predict'
    fname_dataset = '../../../Data/DetectionData_New.h5'
    savepath = '../../../Outputs/'
    indexpath = '../../../Data/TraValIndex_M=100000_Res=0.0_20200812T063630.pkl'
    M = 'all'
    exclude = False
    device = utils.set_device()
    # ==== Checks =============================================================
    if not os.path.exists(fname_dataset):
        raise ValueError('Dataset file not found.')
    if not os.path.exists(indexpath):
        raise ValueError('Index file not found.')
    # =========================================================================
    # Load Data
    # =========================================================================
    if isinstance(M, str) and (M == 'all'):
        M = utils.set_M(fname_dataset, indexpath, exclude=exclude)
    index_tst = utils.set_Tst_index(
        M,
        fname_dataset,
        indexpath,
        exclude=exclude
    )
    # tst_dataset = utils.load_dataset(
    #     fname_dataset,
    #     index_tst,
    #     'True'
    # )
    # =========================================================================
    # Prediction Routine
    # =========================================================================
    initlist = [f'{init_path}/{l}' for l in os.listdir(init_path) if ".ini" in l]
    for init_file in initlist:
        config = configparser.ConfigParser()
        config.read(init_file)
        savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
        parameters = dict(
            fname_dataset=fname_dataset,
            device=device,
            M = M, # Select integer or 'all'
            indexpath=indexpath,
            exclude=exclude,
            batch_size=int(config['PARAMETERS']['batch_size']),
            n_clusters=int(config['PARAMETERS']['n_clusters']),
            savepath=savepath_exp,
            serial=serial_exp,
            show=config['PARAMETERS'].getboolean('show'),
            send_message=config['PARAMETERS'].getboolean('send_message'),
            mode=mode,
            saved_weights=config['PARAMETERS']['saved_weights'],
            max_workers=int(config['PARAMETERS']['max_workers']),
            loaded=True
        )
        utils.save_exp_config(
            savepath_exp,
            serial_exp,
            init_file,
            parameters,
            None
        )
        production.DCM_predict(parameters)
        time.sleep(1)
