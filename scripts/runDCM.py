import sys
sys.path.insert(0, '../RISCluster/')

import importlib as imp
import production
imp.reload(production)
import utils
imp.reload(utils)

if __name__ == '__main__':
    # =========================================================================
    # Universal Parameters
    # =========================================================================
    # Select from 'pretrain', 'train', or 'predict':
    mode = 'pretrain'
    fname_dataset = '../../../Data/DetectionData_New.h5'
    savepath = '../../../Outputs/'
    # Use this for local dev:
    # indexpath = '/Users/williamjenkins/Research/Workflows/RIS_Clustering/Data/TraValIndex_M=100_Res=0.0_20200809T125533.pkl'
    # indexpath = '/Users/williamjenkins/Research/Workflows/RIS_Clustering/Data/TraValIndex_M=500_Res=0.0_20200803T202014.pkl'
    # Use this for full run on Velella:
    # indexpath = '../../../Data/TraValIndex_M=35000_Res=0.0_20200803T212141.pkl'
    indexpath = '../../../Data/TraValIndex_M=100000_Res=0.0_20200812T063630.pkl'
    # Use this for troubleshooting on Velella:
    # indexpath = '../../../Data/TraValIndex_M=1000_Res=0.0_20200803T221100.pkl'
    # =========================================================================
    # Pre-Training Routine
    # =========================================================================
    if mode == 'train':
        savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
        parameters = dict(
            fname_dataset=fname_dataset,
            device=utils.set_device(),
            indexpath=indexpath,
            n_epochs=600,
            savepath=savepath_exp,
            serial=serial_exp,
            show=False,
            send_message=True,
            mode=mode,
            early_stopping=True,
            patience=10
        )
        hyperparameters = dict(
            batch_size=[256, 512, 1024],
            lr=[0.00001, 0.0001, 0.001]
        )
        utils.save_exp_config(
            savepath_exp,
            serial_exp,
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
            n_epochs=200,
            n_clusters=6,
            update_interval=300,
            savepath=savepath_exp,
            serial=serial_exp,
            show=False,
            send_message=True,
            mode=mode,
            saved_weights='../../../Outputs/Models/AEC/Exp20200812T075316/Run_BatchSz=512_LR=0.0001/AEC_Params_20200812T104220.pt'
        )
        hyperparameters = dict(
            n_clusters=[6,7,8],
            batch_size=[512],
            lr=[0.0001],
            gamma=[0.1],
            tol=[0.001]
        )
        # hyperparameters = dict(
        #     batch_size = [128, 256, 512, 1024, 2048],
        #     lr = [0.00001, 0.0001, 0.001],
        #     gamma = [0.08, 0.1, 0.12],
        #     tol = [0.0001, 0.001, 0.01, 0.1]
        # )
        utils.save_exp_config(
            savepath_exp,
            serial_exp,
            parameters,
            hyperparameters
        )
        production.DCM_train(parameters, hyperparameters)
    # =========================================================================
    # Prediction Routine
    # =========================================================================
    if mode == 'predict':
        saved_weights = '../../../Outputs/Models/DCM/Exp20200816T210257/Run_BatchSz=512_LR=0.0001_gamma=0.1_tol=0.001/DCEC_Params_20200816T234307.pt'
        savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
        parameters = dict(
            fname_dataset=fname_dataset,
            device=utils.set_device(),
            M = 'all', # Select integer or 'all'
            indexpath=indexpath,
            exclude=False,
            batch_size=1024,
            n_clusters=5,
            savepath=savepath_exp,
            serial=serial_exp,
            show=False,
            send_message=True,
            mode=mode,
            saved_weights=saved_weights,
            max_workers=14
        )
        utils.save_exp_config(savepath_exp, serial_exp, parameters, None)
        production._predict(parameters)
# End of script.
