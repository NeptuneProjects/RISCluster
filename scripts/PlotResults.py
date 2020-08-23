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

import importlib
import plotting
from processing import get_metadata
importlib.reload(plotting)


def view_cluster_results(csvname, fname_dataset, show=True, save=True, savepath='.'):
    data = genfromtxt(fname, delimiter=',')
    data = np.delete(data,0,0)
    data = data.astype('int')

    label = data[:,0]
    index = data[:,1]
    label_list = np.unique(label)

    for l in range(len(label_list)):
        query = np.where(label == label_list[l])[0]
        N = 16
        image_index = np.random.choice(query, N)

        with h5py.File(fname_dataset, 'r') as f:
            M = len(image_index)
            DataSpec = '/7sec/Spectrogram'
            dset = f[DataSpec]
            fvec = dset[1, 0:64, 0]
            tvec = dset[1, 65, 1:129]
            m, _, _ = dset.shape
            m -= 1
            n = 64
            o = 128
            X = np.empty([M, n, o])
            dset_arr = np.empty([n, o])

            for i in range(M):
                dset_arr = dset[image_index[i], 1:-1, 1:129]
                dset_arr /= dset_arr.max()
                X[i,:,:] = np.expand_dims(dset_arr,axis=0)

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

        fig = plt.figure(figsize=(12,9), dpi=300)
        gs_sup = gridspec.GridSpec(nrows=4, ncols=4, hspace=0.3, wspace=0.3)

        for i in range(N):
            gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_sup[i], hspace=0)

            ax = fig.add_subplot(gs_sub[0])
            plt.imshow(X[i,:,:], extent=extent, aspect='auto', origin='lower')

            tvec = np.linspace(extent[0], extent[1], tr.shape[1])
            ax = fig.add_subplot(gs_sub[1])
            plt.plot(tvec, tr[i,:])

        fig.suptitle(f'Label {label_list[l]}', size=18, weight='bold')
        fig.subplots_adjust(top=0.92)
        if save:
            fig.savefig(f'{savepath}/Label{label_list[l]}_Examples.png')
        if show:
            plt.show()
        else:
            plt.close()

fname = '../../../Outputs/Trials/Exp20200820T225048/Labels20200820T225048.csv'
fname_dataset = '../../../Data/DetectionData_New.h5'
savepath = '../../../Paper/Figures'

view_cluster_results(fname, fname_dataset, show=True, save=False, savepath=savepath)























#
