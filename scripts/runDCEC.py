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
    mode = 'train'
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
    if mode == 'pretrain':
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
        production.DCEC_pretrain(parameters, hyperparameters)
    # =========================================================================
    # Training Routine
    # =========================================================================
    if mode == 'train':
        savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
        parameters = dict(
            fname_dataset=fname_dataset,
            device=utils.set_device(),
            indexpath=indexpath,
            n_epochs=100,
            n_clusters=5,
            update_interval=300,
            savepath=savepath_exp,
            serial=serial_exp,
            show=False,
            send_message=True,
            mode=mode,
            saved_weights='../../../Outputs/Models/AEC/Exp20200812T075316/Run_BatchSz=512_LR=0.0001/AEC_Params_20200812T104220.pt'
        )
        hyperparameters = dict(
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
        production.DCEC_train(parameters, hyperparameters)
    # =========================================================================
    # Prediction Routine
    # =========================================================================
    if mode == 'predict':
        savepath_exp, serial_exp = utils.init_exp_env(mode, savepath)
        parameters = dict(
            fname_dataset=fname_dataset,
            device=utils.set_device(),
            M = 500, # Select integer or 'all'
            indexpath=indexpath,
            exclude=True,
            batch_size=256,
            n_clusters=11,
            savepath=savepath_exp,
            serial=serial_exp,
            show=False,
            send_message=False,
            mode=mode,
            saved_weights='/Users/williamjenkins/Research/Workflows' + \
                '/RIS_Clustering/Outputs/Models/DCEC/Exp20200802T225523' + \
                '/Run_BatchSz=256_LR=0.001_gamma=0.1_tol=0.01/DCEC_Params_ ' +\
                '20200802T225531.pt',
            max_workers=14
        )
        utils.save_exp_config(savepath_exp, serial_exp, parameters, None)
        production.DCEC_predict(parameters)
# End of script.
