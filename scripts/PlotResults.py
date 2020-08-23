import csv
from datetime import datetime
import sys
sys.path.insert(0, '../RISCluster/')

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy import genfromtxt
import torch
from torch.utils.data import DataLoader

import importlib
from networks import DCM
import plotting
from processing import get_metadata
import utils
importlib.reload(plotting)


def view_cluster_results(csvname, fname_dataset, saved_weights, show=True, save=True, savepath='.'):
    device = utils.set_device()
    model = DCM(n_clusters=5).to(device)
    model = utils.load_weights(model, saved_weights, device)

    data = genfromtxt(fname, delimiter=',')
    data = np.delete(data,0,0)
    data = data.astype('int')

    label = data[:,0]
    index = data[:,1]
    label_list = np.unique(label)

    for l in range(len(label_list)):
        query = np.where(label == label_list[l])[0]
        N = 9
        image_index = np.random.choice(query, N)
        metadata = get_metadata(range(N), image_index, fname_dataset)

        dataset = utils.load_dataset(fname_dataset, image_index, send_message=False)
        dataloader = DataLoader(dataset, batch_size=N)
        X = []
        for batch in dataloader:
            X = batch.to(device)

        with h5py.File(fname_dataset, 'r') as f:
            # M = len(image_index)
            DataSpec = '/7sec/Spectrogram'
            dset = f[DataSpec]
            fvec = dset[1, 0:64, 0]
            tvec = dset[1, 65, 1:129]

        # print(X.size())
        # with h5py.File(fname_dataset, 'r') as f:
        #     M = len(image_index)
        #     DataSpec = '/7sec/Spectrogram'
        #     dset = f[DataSpec]
        #     fvec = dset[1, 0:64, 0]
        #     tvec = dset[1, 65, 1:129]
        #     m, _, _ = dset.shape
        #     m -= 1
        #     n = 64
        #     o = 128
        #     X = np.empty([M, n, o])
        #     dset_arr = np.empty([n, o])
        #
        #     for i in range(M):
        #         dset_arr = dset[image_index[i], 1:-1, 1:129]
        #         dset_arr /= dset_arr.max()
        #         X[i,:,:] = np.expand_dims(dset_arr,axis=0)

        with h5py.File(fname_dataset, 'r') as f:
            M = len(image_index)
            DataSpec = '/7sec/Trace'
            dset = f[DataSpec]
            k = 635

            tr = np.empty([M, k])
            dset_arr = np.empty([k,])

            for i in range(M):
                dset_arr = dset[image_index[i], 0:k]
                tr[i,:] = dset_arr/1e-6

        extent = [min(tvec), max(tvec), min(fvec), max(fvec)]

        _, x_r, z = model(X)

        fig = plt.figure(figsize=(12,9), dpi=300)
        gs_sup = gridspec.GridSpec(nrows=int(np.sqrt(N)), ncols=int(np.sqrt(N)), hspace=0.3, wspace=0.3)

        for i in range(N):

            station = metadata[i]['Station']
            try:
                time_on = datetime.strptime(metadata[i]['TriggerOnTime'],
                                            '%Y-%m-%dT%H:%M:%S.%f').strftime(
                                            '%Y-%m-%dT%H:%M:%S.%f')[:-4]
            except:
                time_on = datetime.strptime(metadata[i]['TriggerOnTime'],
                                            '%Y-%m-%dT%H:%M:%S').strftime(
                                            '%Y-%m-%dT%H:%M:%S.%f')[:-4]

            heights = [1, 2, 2]
            widths = [3, 0.2]
            gs_sub = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_sup[i], hspace=0, wspace=0.1, height_ratios=heights, width_ratios=widths)

            tvec = np.linspace(extent[0], extent[1], tr.shape[1])
            ax = fig.add_subplot(gs_sub[0,0])
            plt.plot(tvec, tr[i,:])
            plt.xticks([])
            plt.yticks([])
            plt.title(f'Station {station}; Index: {image_index[i]}\nTrigger: {time_on}', fontsize=10)

            ax = fig.add_subplot(gs_sub[1,0])
            plt.imshow(torch.squeeze(X[i]).detach().numpy(), extent=extent, aspect='auto', origin='lower')
            plt.xticks([])

            ax = fig.add_subplot(gs_sub[2,0])
            plt.imshow(torch.squeeze(x_r[i]).detach().numpy(), extent=extent, aspect='auto', origin='lower')

            ax = fig.add_subplot(gs_sub[:,1])
            plt.imshow(np.expand_dims(z[i].detach().numpy(), 1), cmap='viridis', aspect='auto')
            plt.xticks([])
            plt.yticks([])





        fig.suptitle(f'Label {label_list[l]}', size=18, weight='bold')
        fig.subplots_adjust(top=0.92)
        if save:
            fig.savefig(f'{savepath}/Label{label_list[l]}_Examples.png')
        if show:
            plt.show()
        else:
            plt.close()

fname = '../../../Outputs/Trials/Exp20200823T115040/Labels20200823T115040.csv'
fname_dataset = '../../../Data/DetectionData_New.h5'
savepath = '../../../Paper/Figures'
saved_weights = '/Users/williamjenkins/Research/Workflows/RIS_Clustering/Outputs/Models/DCM/Exp20200823T103757/Run_Clusters=5_BatchSz=512_LR=0.0001_gamma=0.1_tol=0.001/DCM_Params_20200823T103846.pt'

view_cluster_results(fname, fname_dataset, saved_weights, show=True, save=False, savepath=savepath)























#
