import configparser
import csv
from datetime import datetime
import os
import sys
sys.path.insert(0, '../RISCluster/')

import cmocean.cm as cmo
import h5py
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset

from processing import get_metadata
import utils
from networks import AEC, DCM

def cmap_lifeaquatic(N=None):
    """
    Returns colormap inspired by Wes Andersen's The Life Aquatic
    Available from https://jiffyclub.github.io/palettable/wesanderson/
    """
    colors = [
        (27, 52, 108),
        (244, 75, 26),
        (67, 48, 34),
        (35, 81, 53),
        (123, 109, 168),
        (139, 156, 184),
        (214, 161, 66),
        (1, 170, 233),
        (195, 206, 208),
        (229, 195, 158),
    ]
    colors = [tuple([v / 256 for v in c]) for c in colors]
    if colors is not None:
        return colors[0:N]
    else:
        return colors

def centroid_dashboard(z_array, labels, centroids, n_clusters, p=2, show=True):
    d = z_array.shape[1]
    dist_mat = utils.distance_matrix(centroids, centroids, p)
    label_list, counts = np.unique(labels, return_counts=True)
    # vmax = max(centroids.max() / 2, np.median([z_array.max(), centroids.max()]))
    vmax = centroids.max()
    # vmax = z_array.max()
    heights = [0.1 if i==0 else 1 for i in range(1+len(label_list))]
    widths = [3, 2]
    # widths = [0.5 if i==0 else 1 for i in range(1+len(label_list))]
    fig = plt.figure(figsize=(12, 4 * n_clusters), dpi=150)
    gs = gridspec.GridSpec(nrows=1+n_clusters, ncols=2, hspace=0.35, wspace=0.27, height_ratios=heights, width_ratios=widths)

    # Colorbar
    ax = fig.add_subplot(gs[0, :])
    plt.axis('off')
    cmap = 'cmo.deep_r'
    axins = inset_axes(ax, width="50%", height="25%", loc="center")
    norm = mpl.colors.Normalize(vmin=z_array.min(), vmax=vmax)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axins, orientation='horizontal')
    cbar.set_label('Latent Feature Value')

    heights = [3, 2]
    widths = [0.2, 4]
    extent = [0, max(counts), d, 0]

    for l in range(n_clusters):
        distance_d = utils.fractional_distance(centroids[l], z_array, p)
        sort_index_d = np.argsort(distance_d)
        distance_d = distance_d[sort_index_d]
        labels_d = labels[sort_index_d]
        query_i = np.where(labels_d == label_list[l])[0]
        distance_i = distance_d[query_i]
        cdf = np.flip(np.cumsum(np.ones(len(query_i))) / len(query_i))

        labels_not = np.delete(label_list, l)
        centroids_dist = np.delete(dist_mat[l,:], l)
        centroids_ind = np.searchsorted(distance_d, centroids_dist)
        centroids_sortind = np.argsort(centroids_dist)
        centroids_ind = centroids_ind[centroids_sortind]
        labels_not = labels_not[centroids_sortind]

        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[l+1,0], hspace=0, wspace=0, height_ratios=heights, width_ratios=widths)
        # Centroid Plot
        ax = fig.add_subplot(gs_sub[0,0])
        plt.imshow(centroids[l][None].T, cmap=cmap, vmax=vmax)
        plt.xticks([])
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'))
        plt.ylabel('Centroid Feature')
        # Dataset Latent Features
        ax = fig.add_subplot(gs_sub[0,1])
        plt.imshow(z_array[sort_index_d].T, cmap=cmap, aspect='auto', vmax=vmax)
        plt.vlines(centroids_ind, -0.5, 9.5, colors='w', linestyles='dotted')
        for ll in range(n_clusters-1):
            plt.text(centroids_ind[ll], ll+1, str(labels_not[ll]+1), backgroundcolor='w', ha='center', bbox=dict(boxstyle='square,pad=0', facecolor='w', alpha=0.5, edgecolor='w'))
        plt.xticks([])
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'))
        ax.yaxis.tick_right()
        plt.ylabel('Latent Feature')
        ax.yaxis.set_label_position('right')
        plt.title(f"Cluster {l+1}: Dataset")
        # Dataset Distances
        ax = fig.add_subplot(gs_sub[1,1])
        ax.fill_between(query_i, cdf, color="slategray", alpha=0.6, linewidth=0, label=None)
        plt.xlabel('Sorted Sample Index')
        plt.ylim([0, 1.2])
        plt.ylabel('CDF')
        ax2 = ax.twinx()
        ax2.plot(distance_d, c='k', label="Distance")
        plt.vlines(centroids_ind, distance_d.min(), distance_d.max(), colors='b', linestyles='dotted', label="Centroid")
        scttr = plt.scatter(query_i, distance_i, c='firebrick', marker='x', alpha=0.4, label="Member")
        plt.xlim([0, len(z_array)])
        plt.ylim([0, distance_d.max()])
        plt.ylabel('Distance')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        label_offset(ax2, "y")
        if l == 0:
            plt.legend(loc=1, fontsize=6)

        query = np.where(labels == label_list[l])[0]
        z_sub = z_array[query]
        distance = utils.fractional_distance(centroids[l], z_sub, p)
        sort_index = np.argsort(distance)
        distance = distance[sort_index]

        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[l+1,1], hspace=0, wspace=0, height_ratios=heights, width_ratios=widths)
        # Centroid Plot
        ax = fig.add_subplot(gs_sub[0,0])
        plt.imshow(centroids[l][None].T, cmap=cmap, vmax=vmax)
        plt.xticks([])
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'))
        plt.ylabel('Centroid Feature')
        # Cluster Latent Features
        ax = fig.add_subplot(gs_sub[0,1])
        tmp = z_sub.T
        plt.imshow(np.concatenate((tmp, np.zeros((tmp.shape[0], counts.max() - tmp.shape[1]))), axis=1), cmap=cmap, extent=extent, aspect='auto', vmax=vmax)
        plt.xticks([])
        # plt.yticks([])
        plt.yticks(ticks=np.linspace(0,d-1,d)+0.5, labels=np.linspace(1,d,d, dtype='int'))
        ax.yaxis.tick_right()
        plt.ylabel('Latent Feature')
        ax.yaxis.set_label_position('right')
        plt.title(f"Cluster {l+1}: Within Cluster")
        # Cluster Distances
        ax = fig.add_subplot(gs_sub[1,1])
        plt.plot(distance, c='k')
        plt.xlim([0, counts.max()])
        plt.ylim([0, distance_d.max()])
        plt.xlabel('Sorted Sample Index')
        ax.yaxis.tick_right()
        plt.ylabel('Distance')
        ax.yaxis.set_label_position('right')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        label_offset(ax, "y")

    fig.suptitle(fr"L{p} Distance Visualization", size=14)
    fig.subplots_adjust(top=0.96)
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def centroid_distances(z_array, labels, centroids, n_clusters, p=2, show=True):
    dist_mat = utils.distance_matrix(centroids, centroids, p)
    fig = plt.figure(dpi=150)
    plt.imshow(dist_mat, cmap='cmo.solar_r', origin='lower')
    plt.xticks(ticks=np.arange(0, n_clusters), labels=np.arange(1, n_clusters + 1))
    plt.yticks(ticks=np.arange(0, n_clusters), labels=np.arange(1, n_clusters + 1))
    for i in range(n_clusters):
        for j in range(n_clusters):
            plt.text(i, j, f"{dist_mat[i,j]:.1f}", backgroundcolor='w', ha='center', bbox=dict(boxstyle='square,pad=0', facecolor='w', edgecolor='w'))
    cbar = plt.colorbar()
    cbar.set_label('Distance')
    fig.suptitle(f"L-{p} Distance Matrix", size=14)
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def cluster_gallery(
        model,
        dataset,
        fname_dataset,
        index_tra,
        device,
        z_array,
        labels,
        centroids=None,
        p=2,
        show=True
    ):
    model.eval()
    label_list, counts = np.unique(labels, return_counts=True)
    if centroids is not None:
        X_c = model.decoder(torch.from_numpy(centroids).float().to(device))
    else:
        centroids = model.clustering.weights
        X_c = model.decoder(centroids)
        centroids = centroids.detach().cpu().numpy()
    N = 4
    fig = plt.figure(figsize=(len(label_list),8), dpi=150)
    heights = [1 for i in range(N+1)] + [0.2]
    gs_sup = gridspec.GridSpec(nrows=N+2, ncols=len(label_list), hspace=0.1, wspace=0.1, height_ratios=heights)
    heights = [1, 4, 0.5]
    font = {
        'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 5,
        }
    transform = 'sample_norm_cent'
    cmap_feat = cmo.deep_r
    cmap_spec = cmo.dense
    vmax = centroids.max()
    for l, label in enumerate(label_list):

        query = np.where(labels == label)[0]
        z_sub = z_array[query]
        distance = utils.fractional_distance(centroids[l], z_sub, p)
        sort_index = np.argsort(distance)[0:N]
        load_index = query[sort_index]
        distance = distance[sort_index]

        subset = Subset(dataset, load_index)
        dataloader = DataLoader(subset, batch_size=N)

        for batch in dataloader:
            idx, X = batch
            idx.numpy()
            X.to(device)
            with h5py.File(fname_dataset, 'r') as f:
                M = len(idx)
                DataSpec = '/4s/Trace'
                dset = f[DataSpec]
                k = 351

                tr = np.empty([M, k])
                dset_arr = np.empty([k,])

                for i in range(M):
                    dset_arr = dset[idx[i], 25:-25]
                    tr[i,:] = dset_arr

        Z = model.encoder(X)

        gs_sub = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_sup[0,l], hspace=0, wspace=0, height_ratios=heights)
        ax = fig.add_subplot(gs_sub[0])
        plt.axis('off')
        ax = fig.add_subplot(gs_sub[1])
        plt.imshow(torch.squeeze(X_c[l]).detach().cpu().numpy(), cmap=cmap_spec, aspect='auto', origin='lower')
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(f"Cluster {label+1}", size=10)
        if l == 0:
            plt.ylabel("Centroids", size=10)

        ax = fig.add_subplot(gs_sub[2])
        plt.imshow(np.expand_dims(centroids[l], 0), cmap=cmap_feat, aspect='auto', vmax = vmax)
        plt.xticks([])
        plt.yticks([])

        for i in range(N):
            gs_sub = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_sup[i+1,l], hspace=0, wspace=0, height_ratios=heights)
            ax = fig.add_subplot(gs_sub[0])
            plt.plot(tr[i], 'k', linewidth=0.5)
            plt.xlim((0, k))
            plt.xticks([])
            plt.yticks([])

            ax = fig.add_subplot(gs_sub[1])
            plt.imshow(np.squeeze(X[i,:,:].detach().cpu().numpy()), cmap=cmap_spec, aspect='auto', origin='lower')
            # plt.text(0, 60, f"{load_index[i]}", fontdict=font)
            plt.text(110, 60, f"d={distance[i]:.1f}", fontdict=font)
            plt.xticks([])
            plt.yticks([])

            ax = fig.add_subplot(gs_sub[2])
            plt.imshow(np.expand_dims(Z[i].detach().cpu().numpy(), 0), cmap=cmap_feat, aspect='auto', vmax = vmax)
            plt.xticks([])
            plt.yticks([])

    gs_sub = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_sup[-1,:])
    # Colorbar: Specgram
    ax = fig.add_subplot(gs_sub[0])
    plt.axis('off')
    cmap = 'cmo.deep_r'
    axins = inset_axes(ax, width="75%", height="50%", loc="center")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_spec), cax=axins, orientation='horizontal')
    cbar.set_label('Normalized Spectrogram Value')

    # Colorbar: Latent Space
    ax = fig.add_subplot(gs_sub[1])
    plt.axis('off')
    cmap = 'cmo.deep_r'
    axins = inset_axes(ax, width="75%", height="50%", loc="center")
    norm = mpl.colors.Normalize(vmin=z_array.min(), vmax=vmax)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_feat), cax=axins, orientation='horizontal')
    cbar.set_label('Latent Feature Value')

    if show:
        plt.show()
    else:
        plt.close()
    return fig

def compare_images(
        model,
        epoch,
        disp_idx,
        fname_dataset,
        savepath=None,
        show=True,
        mode='multi'
    ):
    images, tvec, fvec = utils.load_images(fname_dataset, disp_idx)
    disp_loader = DataLoader(
        utils.SeismoDataset(images),
        batch_size=len(disp_idx)
    )
    disp = next(iter(disp_loader))

    model.eval()
    x_r, z = model(disp)
    figtitle = f'Pre-training: Epoch {epoch}'
    n, o = list(disp.size()[2:])
    if mode == 'multi':
        fig = view_specgram_training(
            disp,
            x_r, z, n, o,
            figtitle,
            disp_idx,
            tvec,
            fvec,
            fname_dataset,
            show=show
        )
        if savepath is not None:
            savepath_snap = savepath + '/snapshots/'
            if not os.path.exists(savepath_snap):
                os.makedirs(savepath_snap)
            figname = savepath_snap + f'AEC_Training_Epoch_{epoch:03d}.png'
            fig.savefig(figname, dpi=300)
    elif mode == 'single':
        fig = view_specgram_training2(
            disp,
            x_r, z, n, o,
            figtitle,
            disp_idx,
            tvec,
            fvec,
            fname_dataset,
            show=show
        )
    return fig

def label_offset(ax, axis="y"):
    if axis == "y":
        fmt = ax.yaxis.get_major_formatter()
        ax.yaxis.offsetText.set_visible(False)
        set_label = ax.set_ylabel
        label = ax.get_ylabel()

    elif axis == "x":
        fmt = ax.xaxis.get_major_formatter()
        ax.xaxis.offsetText.set_visible(False)
        set_label = ax.set_xlabel
        label = ax.get_xlabel()

    def update_label(event_axes):
        offset = fmt.get_offset()
        if offset == '':
            set_label("{}".format(label))
        else:
            set_label("{} ({})".format(label, offset))
        return

    ax.callbacks.connect("ylim_changed", update_label)
    ax.callbacks.connect("xlim_changed", update_label)
    ax.figure.canvas.draw()
    update_label(None)
    return

def save_DCM_output(x, label, x_rec, z, idx, savepath):
    # print(f'x={type(x)} | label={type(label)} | x_r={type(x_rec)} | z={type(z)} | idx={type(idx)} | path={type(savepath)}')
    fig = view_DCM_output(x, label, x_rec, z, idx, show=False)
    # print(f'{savepath}{idx:07d}.png')
    fig.savefig(f'{savepath}/{idx:07d}.png')
    return None

def view_centroid_output(centroids, X_r, figtitle, show=True):
    '''Reconstructs spectrograms from cluster centroids.'''
    n, o = list(X_r.size())[2:]
    widths = [2, 0.1]
    heights = [1 for i in range(len(centroids))]

    fig = plt.figure(figsize=(3,2*len(centroids)), dpi=150)
    gs = gridspec.GridSpec(nrows=len(centroids), ncols=2, hspace=0.5, wspace=0.1, width_ratios=widths)

    for i in range(len(centroids)):
        ax = fig.add_subplot(gs[i,0])
        plt.imshow(torch.reshape(X_r[i,:,:,:], (n,o)).cpu().detach().numpy(), aspect='auto')
        plt.gca().invert_yaxis()
        plt.title(f'Cluster {i}')

        ax = fig.add_subplot(gs[i,1])
        plt.imshow(np.expand_dims(centroids[i].cpu().detach().numpy(), 1), cmap='viridis', aspect='auto')
        plt.xticks([])
        plt.yticks([])

    fig.suptitle(figtitle, size=14)
    fig.subplots_adjust(top=0.93)
    if show is False:
        plt.close()
    else:
        plt.show()
    return fig

def view_class_cdf(
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        n_clusters,
        p=2,
        show=True
    ):
    label_list, counts_a = np.unique(labels_a, return_counts=True)
    _, counts_b = np.unique(labels_b, return_counts=True)

    fig = plt.figure(figsize=(6, 2*int(np.ceil(n_clusters/2))), dpi=100)
    gs = gridspec.GridSpec(nrows=int(np.ceil(n_clusters/2)), ncols=2, hspace=0.9, wspace=0.4)
    colors = cmap_lifeaquatic(n_clusters)

    for l in range(n_clusters):
        ax = fig.add_subplot(gs[l])

        distance_a = utils.fractional_distance(centroids_a[l], data_a, p)
        sort_index_a = np.argsort(distance_a)
        distance_a = distance_a[sort_index_a]
        labels_a_ = labels_a[sort_index_a]
        query_a = np.where(labels_a_ == label_list[l])[0]
        distance_a = distance_a[query_a]
        cdf_a = np.cumsum(np.ones(len(query_a))) / len(query_a)

        distance_b = utils.fractional_distance(centroids_b[l], data_b, p)
        sort_index_b = np.argsort(distance_b)
        distance_b = distance_b[sort_index_b]
        labels_b_ = labels_b[sort_index_b]
        query_b = np.where(labels_b_ == label_list[l])[0]
        distance_b = distance_b[query_b]
        cdf_b = np.cumsum(np.ones(len(query_b))) / len(query_b)

        plt.plot(distance_a, cdf_a, color=colors[0])
        plt.plot(distance_b, cdf_b, color=colors[1])

        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsbsy}')
        plt.xlabel(f"$d=\Vert\mathbf{{r}} - \pmb{{\mu}}_{l+1}\Vert_{p}$")
        plt.ylabel(f"$F_{l+1}(d)$", rotation=90, ha="center")
        plt.title(f"Class {l + 1} CDF")

    if show:
        plt.show()
    else:
        plt.close()
    return fig

def view_class_pdf(
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        n_clusters,
        p=2,
        show=True
    ):
    label_list, counts_a = np.unique(labels_a, return_counts=True)
    _, counts_b = np.unique(labels_b, return_counts=True)
    dx = 1
    nbins = 100
    X = np.linspace(0, 301, nbins)

    fig = plt.figure(figsize=(12, 2*int(np.ceil(n_clusters/2))), dpi=100)
    gs = gridspec.GridSpec(nrows=int(np.ceil(n_clusters/2)), ncols=2, hspace=1, wspace=0.25)
    colors = cmap_lifeaquatic(n_clusters)
    for l in range(n_clusters):
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[l], hspace=0, wspace=0)

        distance_a = utils.fractional_distance(centroids_a[l], data_a, p)
        sort_index_a = np.argsort(distance_a)
        distance_a = distance_a[sort_index_a]
        labels_a_ = labels_a[sort_index_a]
        distance_b = utils.fractional_distance(centroids_b[l], data_b, p)
        sort_index_b = np.argsort(distance_b)
        distance_b = distance_b[sort_index_b]
        labels_b_ = labels_b[sort_index_b]
        # max_dist = np.max([distance_a.max(), distance_b.max()])

        axa = fig.add_subplot(gs_sub[0])
        for ll in range(n_clusters):
            query_a = np.where(labels_a_ == label_list[ll])[0]
            distance_a_ = distance_a[query_a]
            hist_a = np.histogram(distance_a_, bins=X, density=True)[0]

            plt.plot(X[:-1], hist_a, color=colors[ll], label=f"{ll+1}")
            plt.fill_between(X[:-1], 0, hist_a, color=colors[ll], alpha=0.2)
            plt.xlim(X.min(), X.max())
            plt.xticks([])
            plt.ylim(0.001, 1)
            plt.yscale('log')
            plt.yticks([0.001, 0.01, 0.1])
            plt.text(1, 0.9, 'Before DEC', ha='right', va='top', transform=axa.transAxes)
            plt.rc('text', usetex=True)
            plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsbsy}')
            plt.ylabel(f"$f_j(d)$", rotation=0, ha="right", y=-0.2, size=14)
            # plt.ylabel(f"$\mathrm{{PDF}}_j(\Vert\mathbf{{r}} - \pmb{{\mu}}_{l+1}\Vert_{p})$", rotation=90, ha="center", y=-0.1, size=14)
            plt.title(f"Class PDFs relative to $\pmb{{\mu}}_{l+1}$", loc="left", size=14)

        axb = fig.add_subplot(gs_sub[1])
        for ll in range(n_clusters):
            query_b = np.where(labels_b_ == label_list[ll])[0]
            distance_b_ = distance_b[query_b]
            hist_b = np.histogram(distance_b_, bins=X, density=True)[0]

            plt.plot(X[:-1], hist_b, color=colors[ll], label=f"{ll+1}")
            plt.fill_between(X[:-1], 0, hist_b, color=colors[ll], alpha=0.2)
            plt.xlim(X.min(), X.max())
            plt.rc('text', usetex=True)
            plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsbsy}')
            plt.xlabel(f"$d=\Vert\mathbf{{r}} - \pmb{{\mu}}_{l+1}\Vert_{p}$", size=14)
            plt.ylim(0.001, 1)
            plt.yscale('log')
            plt.yticks(ticks=[0.001, 0.01, 0.1])
            plt.text(1, 0.9, 'After DEC', ha='right', va='top', transform=axb.transAxes)

    fig.add_axes([0.9, 0.67, 0.1, 0.1]).axis('off')
    handles, labels = axa.get_legend_handles_labels()
    leg = plt.legend(handles, labels, fontsize=14, loc="right")
    leg.set_title("Classes", prop = {'size':'x-large'})

    if show:
        plt.show()
    else:
        plt.close()
    return fig

def view_cluster_results(exppath, show=True, save=True, savepath='.'):
    init_file = [f for f in os.listdir(exppath) if f.endswith('.ini')][0]
    init_file = f'{exppath}/{init_file}'
    config = configparser.ConfigParser()
    config.read(init_file)
    fname_dataset = config['UNIVERSAL']['fname_dataset']
    transform = config['PARAMETERS']['transform']
    AEC_weights = config['PARAMETERS']['saved_weights']
    n_clusters = int(config['PARAMETERS']['n_clusters'])
    DCM_weights_path = f"{config['UNIVERSAL']['savepath']}Models/DCM/{exppath.split('/')[-2]}/{exppath.split('/')[-1]}"
    DCM_weights = [f for f in os.listdir(DCM_weights_path) if f.endswith('.pt')][0]
    DCM_weights = f"{DCM_weights_path}/{DCM_weights}"
    label, index, label_list = utils.load_labels(exppath)
    # device = utils.set_device()
    device = 'cpu'
    aec = AEC().to(device)
    aec = utils.load_weights(aec, AEC_weights, device)
    dcm = DCM(n_clusters=n_clusters).to(device)
    dcm = utils.load_weights(dcm, DCM_weights, device)
    centroids = dcm.clustering.weights
    X_c = dcm.decoder(centroids)

    for l in range(len(label_list)):
        query = np.where(label == label_list[l])[0]
        N = 8
        image_index = np.random.choice(query, N, replace=False)
        metadata = get_metadata(range(N), image_index, fname_dataset)

        dataset = utils.load_dataset(fname_dataset, image_index, send_message=False, transform=transform)
        dataloader = DataLoader(dataset, batch_size=N)
        X = []
        for batch in dataloader:
            X = batch.to(device)

        with h5py.File(fname_dataset, 'r') as f:
            # M = len(image_index)
            DataSpec = '/4s/Spectrogram'
            dset = f[DataSpec]
            # fvec = dset[1, 0:64, 0]
            fvec = dset[1, 0:68, 0]
            # tvec = dset[1, 65, 12:-14]
            tvec = dset[1, 69, 12:-14]
            # tvec = dset[1, 65, :]

        with h5py.File(fname_dataset, 'r') as f:
            M = len(image_index)
            DataSpec = '/4s/Trace'
            dset = f[DataSpec]
            k = 351

            tr = np.empty([M, k])
            dset_arr = np.empty([k,])

            for i in range(M):
                dset_arr = dset[image_index[i], 25:-25]
                tr[i,:] = dset_arr/1e-6

        extent = [min(tvec), max(tvec), min(fvec), max(fvec)]
        x_r_pretrain, z_pretrain = aec(X)
        _, x_r_train, z_train = dcm(X)

        fig = plt.figure(figsize=(12,9), dpi=150)
        gs_sup = gridspec.GridSpec(nrows=int(np.sqrt(N)+1), ncols=int(np.sqrt(N)+1), hspace=0.3, wspace=0.3)

        widths = [4, 0.2]
        gs_sub = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_sup[0], hspace=0, wspace=0.1, width_ratios=widths)

        ax = fig.add_subplot(gs_sub[0])
        plt.imshow(torch.squeeze(X_c[l]).detach().cpu().numpy(), extent=extent, aspect='auto', origin='lower')
        plt.xticks([])
        plt.xlabel('Time (s)', size=7)
        plt.ylabel('Frequency (Hz)', size=7)
        plt.title('Centroid Reconstruction', fontsize=8)

        ax = fig.add_subplot(gs_sub[1])
        plt.imshow(np.expand_dims(centroids[l].detach().cpu().numpy(), 1), cmap='viridis', aspect='auto')
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('Centroid', size=5, rotation=90)


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

            heights = [1, 3, 3, 3]
            widths = [4, 0.2, 0.2]
            gs_sub = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs_sup[i+1], hspace=0, wspace=0.1, height_ratios=heights, width_ratios=widths)

            tvec = np.linspace(extent[0], extent[1], tr.shape[1])
            ax = fig.add_subplot(gs_sub[0,0])
            plt.plot(tvec, tr[i,:])
            plt.xticks([])
            plt.yticks([])
            plt.title(f'Station {station}; Index: {image_index[i]}\nTrigger: {time_on}', fontsize=8)

            ax = fig.add_subplot(gs_sub[1,0])
            plt.imshow(torch.squeeze(X[i]).detach().cpu().numpy(), extent=extent, aspect='auto', origin='lower')
            plt.xticks([])
            plt.ylabel('Original', size=7)

            ax = fig.add_subplot(gs_sub[2,0])
            plt.imshow(torch.squeeze(x_r_pretrain[i]).detach().cpu().numpy(), extent=extent, aspect='auto', origin='lower')
            plt.ylabel('Pre-trained', size=7)

            ax = fig.add_subplot(gs_sub[3,0])
            plt.imshow(torch.squeeze(x_r_train[i]).detach().cpu().numpy(), extent=extent, aspect='auto', origin='lower')
            plt.ylabel('Trained', size=7)

            ax = fig.add_subplot(gs_sub[1:,1])
            plt.imshow(np.expand_dims(z_pretrain[i].detach().cpu().numpy(), 1), cmap='viridis', aspect='auto')
            plt.xticks([])
            plt.yticks([])
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('Pre-trained', size=5, rotation=90)

            ax = fig.add_subplot(gs_sub[1:,2])
            plt.imshow(np.expand_dims(z_train[i].detach().cpu().numpy(), 1), cmap='viridis', aspect='auto')
            plt.xticks([])
            plt.yticks([])
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('Trained', size=5, rotation=90)

        fig.suptitle(f'Label {label_list[l]}', size=14)
        fig.subplots_adjust(top=0.91)
        if show:
            plt.show()
        else:
            plt.close()
        if save:
            print(f'{savepath}/Label{label_list[l]:02d}_Examples.png')
            fig.savefig(f'{savepath}/Label{label_list[l]:02d}_Examples.png', dpi=300)

def view_cluster_stats(k_list, inertia, silh, gap_g, gap_u, show=False):
    def _make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # fig = plt.figure(figsize=(12,9), dpi=300)
    fig, host = plt.subplots(figsize=(6,4), dpi=300)

    par1 = host.twinx()
    par2 = host.twinx()

    par2.spines["right"].set_position(("axes", 1.25))
    _make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)

    p1, = host.plot(inertia, color="navy", marker=".", label="Inertia")
    p2, = par1.plot(silh, color="darkgreen", marker=".", label="Silhouette")
    p3, = par2.plot(gap_g, "firebrick", ls=":", marker=".", label="Gaussian")
    p4, = par2.plot(gap_u, "firebrick", ls="-.", marker=".", label="Uniform")

    host.set_xlabel("Number of Clusters")
    host.set_ylabel("Inertia")
    par1.set_ylabel("Silhouette Score")
    par2.set_ylabel("Gap Statistic")

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p4.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)
    host.set_xticks(range(len(k_list)))
    host.set_xticklabels(k_list)
    for label in host.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)

    lines = [p1, p2, p3, p4]
    host.xaxis.grid()
    host.set_xlim(k_list[0]-3, k_list[-1]-1)
    leg = host.legend(lines, [l.get_label() for l in lines], ncol=4, bbox_to_anchor=(0.6, -0.28), loc='lower center')
    # plt.title("K-Means Metrics")
    plt.tight_layout()
    plt.subplots_adjust(right=0.7, bottom=0.2)
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def view_DCM_output(x, label, x_rec, z, idx, figsize=(12,9), show=False):
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1,0.1,1])
    # Original Spectrogram
    ax = fig.add_subplot(gs[0])
    plt.imshow(np.squeeze(x), aspect='auto')
    plt.ylabel('Frequency Bin')
    plt.xlabel('Time Bin')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Original Spectrogram')
    # Latent Space Representation:
    ax = fig.add_subplot(gs[1])
    plt.imshow(np.expand_dims(z, 1), cmap='viridis', aspect='auto')
    plt.gca().invert_yaxis()
    plt.title('Latent Space')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False  # labels along the bottom edge are off
    )
    # Reconstructed Spectrogram:
    ax = fig.add_subplot(gs[2])
    plt.imshow(np.squeeze(x_rec), aspect='auto')
    plt.ylabel('Frequency Bin')
    plt.xlabel('Time Bin')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Reconstructed Spectrogram')
    fig.suptitle(f'Cluster: {label}', size=18, weight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if show is False:
        plt.close()
    else:
        plt.show()
    return fig

def view_detections(fname_dataset, image_index, figtitle,
                  nrows=2, ncols=2, figsize=(12,9), show=True):
    sample_index = np.arange(0, len(image_index))

    with h5py.File(fname_dataset, 'r') as f:
        M = len(image_index)
        DataSpec = '/4s/Spectrogram'
        dset = f[DataSpec]
        # fvec = dset[1, 0:64, 0]
        fvec = dset[1, 0:68, 0]
        # tvec = dset[1, 65, 12:-14]
        tvec = dset[1, 69, 12:-14]
        m, _, _ = dset.shape
        m -= 1
        # n = 65
        n = 69
        o = 175
        X = np.empty([M, n, o])
        dset_arr = np.empty([n, o])

        for i in range(M):
            dset_arr = dset[image_index[i], :-1, 12:-14]
            dset_arr /= dset_arr.max()
            X[i,:,:] = dset_arr

    with h5py.File(fname_dataset, 'r') as f:
        M = len(image_index)
        DataSpec = '/4s/Trace'
        dset = f[DataSpec]
        k = 635

        tr = np.empty([M, k])
        dset_arr = np.empty([k,])

        for i in range(M):
            dset_arr = dset[image_index[i], 0:k]
            tr[i,:] = dset_arr/1e-6

    extent = [min(tvec), max(tvec), min(fvec), max(fvec)]
    '''Plots selected spectrograms from input data.'''
    if not len(sample_index) == nrows * ncols:
        raise ValueError('Subplot/sample number mismatch: check dimensions.')
    metadata = get_metadata(sample_index, image_index, fname_dataset)
    fig = plt.figure(figsize=figsize, dpi=300)
    gs_sup = gridspec.GridSpec(nrows=nrows, ncols=ncols, hspace=0.4, wspace=0.25)
    counter = 0
    for i in range(len(sample_index)):
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_sup[i], hspace=0)

        ax = fig.add_subplot(gs_sub[0])
        plt.imshow(X[sample_index[i],:,:], extent=extent, aspect='auto', origin='lower')
        ax.set_xticks([])
        plt.ylabel('Frequency (Hz)')
        station = metadata[counter]['Station']
        try:
            time_on = datetime.strptime(metadata[counter]['TriggerOnTime'],
                                        '%Y-%m-%dT%H:%M:%S.%f').strftime(
                                        '%Y-%m-%dT%H:%M:%S.%f')[:-4]
        except:
            time_on = datetime.strptime(metadata[counter]['TriggerOnTime'],
                                        '%Y-%m-%dT%H:%M:%S').strftime(
                                        '%Y-%m-%dT%H:%M:%S.%f')[:-4]
        plt.title(f'Station {station}\nTrigger: {time_on}; '
                  f'Index: {image_index[sample_index[i]]}')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(cax=cax)

        tvec = np.linspace(extent[0], extent[1], tr.shape[1])

        ax = fig.add_subplot(gs_sub[1])
        plt.plot(tvec, tr[i,:])
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (1e-6 m/s)')

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cax.axis('off')

        counter += 1
    fig.suptitle(figtitle, size=18, weight='bold')
    # fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def view_latent_space(
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        n_clusters,
        p,
        show=True
    ):

    d = data_a.shape[1]
    dist_mat_a = utils.distance_matrix(centroids_a, centroids_a, p)
    dist_mat_b = utils.distance_matrix(centroids_b, centroids_b, p)
    label_list, counts_a = np.unique(labels_a, return_counts=True)
    _, counts_b = np.unique(labels_b, return_counts=True)
    vmin = np.min([centroids_a.min(), centroids_b.min()])
#     vmin = 0
    vmax = np.max([centroids_a.max(), centroids_b.max()])
    nrows = int(np.ceil(n_clusters/2))
    heights = [1 for i in range(nrows)]

    fig = plt.figure(figsize=(8, 2.5*nrows), dpi=150)
    gs = gridspec.GridSpec(nrows=nrows, ncols=2, height_ratios=heights, hspace=0.3, wspace=0.05)
    cmap = 'cmo.deep_r'
    widths = [0.5, 4]

    for l in range(n_clusters):
        distance_d = utils.fractional_distance(centroids_a[l], data_a, p)
        sort_index_d = np.argsort(distance_d)
        distance_d = distance_d[sort_index_d]
        labels_d = labels_a[sort_index_d]
        query_i = np.where(labels_d == label_list[l])[0]
        distance_i = distance_d[query_i]

        labels_not = np.delete(label_list, l)
        centroids_dist = np.delete(dist_mat_a[l,:], l)
        centroids_ind = np.searchsorted(distance_d, centroids_dist)
        centroids_sortind = np.argsort(centroids_dist)
        centroids_ind = centroids_ind[centroids_sortind]
        labels_not = labels_not[centroids_sortind]

        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[l], hspace=0.04, wspace=0, width_ratios=widths)

        # Centroid A
        ax = fig.add_subplot(gs_sub[0,0])
        plt.imshow(centroids_a[l][None].T, cmap=cmap, vmax=vmax)
        plt.xticks([])
        if l == 0:
            plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'), size=5)
            plt.ylabel('Before DEC', size=12, y=0.6)
            plt.rc('text', usetex=True)
            plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsbsy}')
            plt.title(f"$\pmb{{\mu}}_j$", size=14)
        else:
            plt.yticks(ticks=np.linspace(0,d-1,d), labels=[], size=5)

        # Latent Space A
        ax = fig.add_subplot(gs_sub[0,1])
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsbsy}')
        plt.imshow(data_a[sort_index_d].T, cmap=cmap, aspect='auto', vmax=vmax)
        plt.vlines(centroids_ind, -0.5, 9.5, colors='w', ls='dashed', lw=0.75, alpha=0.5)
        for ll in range(n_clusters-1):
            plt.text(centroids_ind[ll], 1.2*(ll+1), f"$\pmb{{\mu}}_{labels_not[ll]+1}$", size=6, backgroundcolor='w', ha='center', bbox=dict(boxstyle='square,pad=0', facecolor='w', alpha=1, edgecolor='w'))
        plt.xticks([])
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=[])
        if l == 0:
            plt.text(0.03, 1.1, f"$\pmb{{z}}_i \in Z$", size=14, transform=ax.transAxes)
        plt.title(f"$j={l+1}$", size=14)

        distance_d = utils.fractional_distance(centroids_b[l], data_b, p)
        sort_index_d = np.argsort(distance_d)
        distance_d = distance_d[sort_index_d]
        labels_d = labels_b[sort_index_d]
        query_i = np.where(labels_d == label_list[l])[0]
        distance_i = distance_d[query_i]

        labels_not = np.delete(label_list, l)
        centroids_dist = np.delete(dist_mat_b[l,:], l)
        centroids_ind = np.searchsorted(distance_d, centroids_dist)
        centroids_sortind = np.argsort(centroids_dist)
        centroids_ind = centroids_ind[centroids_sortind]
        labels_not = labels_not[centroids_sortind]

        # Centroid B
        ax = fig.add_subplot(gs_sub[1,0])
        plt.imshow(centroids_b[l][None].T, cmap=cmap, vmax=vmax)
        plt.xticks([])
        if l == 0:
            plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'), size=5)
            plt.ylabel('After DEC', size=12, y=0.4)
        else:
            plt.yticks(ticks=np.linspace(0,d-1,d), labels=[], size=5)

        # Latent Space B
        ax = fig.add_subplot(gs_sub[1,1])
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsbsy}')
        plt.imshow(data_b[sort_index_d].T, cmap=cmap, aspect='auto', vmax=vmax)
        plt.vlines(centroids_ind, -0.5, 9.5, colors='w', ls='dashed', lw=0.75, alpha=0.5)
        for ll in range(n_clusters-1):
            plt.text(centroids_ind[ll], 1.2*(ll+1), f"$\pmb{{\mu}}_{labels_not[ll]+1}$", size=6, backgroundcolor='w', ha='center', bbox=dict(boxstyle='square,pad=0', facecolor='w', alpha=1, edgecolor='w'))
        if l == 0:
            label = ax.set_xlabel("$i$", size=14)
            ax.xaxis.set_label_coords(-0.03, 0)
        else:
            xlabels = [item.get_text() for item in ax.get_xticklabels()]
            empty_string_labels = ['']*len(xlabels)
            ax.set_xticklabels(empty_string_labels)
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=[])

    # Colorbar
    ax = fig.add_axes([0, 0.05, 1, 0.1])
    plt.axis('off')
    axins = inset_axes(ax, width="50%", height="15%", loc="center")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axins, orientation='horizontal')
    cbar.set_label('Latent Feature Value', size=14)

    fig.suptitle(f"Latent space sorted by $d_{{i,j}}=\Vert \pmb{{r}}_i-\pmb{{\mu}}_j \Vert_{p} \mid d_{{i+1,j}} > d_{{i,j}}$", size=18)
    fig.subplots_adjust(top=0.91)

    if show:
        plt.show()
    else:
        plt.close()
    return fig

def view_learningcurve(training_history, validation_history, show=True):
    epochs = len(training_history['mse'])
    fig = plt.figure(figsize=(18,6), dpi=300)
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax = fig.add_subplot(gs[0])
    plt.plot(range(epochs), training_history['mse'], label='Training')
    plt.plot(range(epochs), validation_history['mse'], label='Validation')
    plt.xlabel('Epochs', size=14)
    plt.ylabel('MSE', size=14)
    plt.title('Loss: Mean Squared Error', weight='bold', size=18)
    plt.legend()

    ax = fig.add_subplot(gs[1])
    plt.plot(range(epochs), training_history['mae'], label='Training')
    plt.plot(range(epochs), validation_history['mae'], label='Validation')
    plt.xlabel('Epochs', size=14)
    plt.ylabel('MAE', size=14)
    plt.title('Loss: Mean Absolute Error', weight='bold', size=18)
    plt.legend()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def view_PCA(pca2d, labels):
    fig = plt.figure(figsize=(6,6), dpi=100)
    sns.scatterplot(pca2d[:,0], pca2d[:,1], hue=labels, palette='Set1', alpha=0.2)
    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    fig.tight_layout()
    return fig

def view_specgram(X, insp_idx, n, o, fname_dataset, sample_index, figtitle,
                  nrows=2, ncols=2, figsize=(12,9), show=True):
    '''Plots selected spectrograms from input data.'''
    if not len(insp_idx) == nrows * ncols:
        raise ValueError('Subplot/sample number mismatch: check dimensions.')
    metadata = get_metadata(insp_idx, sample_index, fname_dataset)
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    counter = 0
    for i in range(len(insp_idx)):

        # starttime = metadata[counter]['StartTime']
        # npts = int(metadata[counter]['Npts'])
        # freq = str(1000 * metadata[counter]['SamplingInterval']) + 'ms'
        # tvec = pd.date_range(starttime, periods=npts, freq=freq)
        # print(tvec)

        ax = fig.add_subplot(gs[i])
        plt.imshow(torch.reshape(X[insp_idx[i],:,:,:], (n,o)), aspect='auto')
        plt.gca().invert_yaxis()
        plt.xlabel('Time Bin')
        plt.ylabel('Frequency Bin')
        station = metadata[counter]['Station']
        try:
            time_on = datetime.strptime(metadata[counter]['TriggerOnTime'],
                                        '%Y-%m-%dT%H:%M:%S.%f').strftime(
                                        '%Y-%m-%dT%H:%M:%S.%f')[:-4]
        except:
            time_on = datetime.strptime(metadata[counter]['TriggerOnTime'],
                                        '%Y-%m-%dT%H:%M:%S').strftime(
                                        '%Y-%m-%dT%H:%M:%S.%f')[:-4]
        plt.title(f'Station {station}\nTrigger: {time_on}\n'
                  f'Index: {sample_index[insp_idx[i]]}')
        # plt.title(f'Station {}'.format(metadata[counter]['Station']))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        counter += 1
    fig.suptitle(figtitle, size=18, weight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def view_specgram_training(
        x, x_r, z, n, o,
        figtitle,
        disp_idx,
        tvec,
        fvec,
        fname_dataset,
        figsize=(6,5),
        show=True
    ):
    sample_idx = np.arange(0, len(disp_idx))
    metadata = get_metadata(sample_idx, disp_idx, fname_dataset)
    X = x.detach().cpu().numpy()
    X_r = x_r.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    fig = plt.figure(figsize=figsize, dpi=100)
    heights = [4, 0.4, 4]
    extent = [min(tvec), max(tvec), min(fvec), max(fvec)]
    gs = gridspec.GridSpec(nrows=3, ncols=4, height_ratios=heights, wspace=0.3)
    counter = 0
    for i in range(x.size()[0]):
        station = metadata[i]['Station']
        try:
            time_on = datetime.strptime(metadata[i]['TriggerOnTime'],
                                        '%Y-%m-%dT%H:%M:%S.%f').strftime(
                                        '%Y-%m-%dT%H:%M:%S.%f')[:-4]
        except:
            time_on = datetime.strptime(metadata[i]['TriggerOnTime'],
                                        '%Y-%m-%dT%H:%M:%S').strftime(
                                        '%Y-%m-%dT%H:%M:%S.%f')[:-4]

        ax = fig.add_subplot(gs[0,counter])
        plt.imshow(np.reshape(X[i,:,:,:], (n,o)), cmap='cmo.ice_r', extent=extent, aspect='auto', origin='lower')
        plt.xlabel('Time (s)')
        if i == 0:
            plt.ylabel('Frequency (Hz)')
        plt.title(f'Station {station}; Index: {disp_idx[i]}\nTrigger: {time_on}', fontsize=8)

        ax = fig.add_subplot(gs[1,counter])
        plt.imshow(np.expand_dims(z[i], 0), cmap=cmo.deep_r, aspect='auto')
        plt.xticks([])
        plt.yticks([])

        ax = fig.add_subplot(gs[2,counter])
        plt.imshow(np.reshape(X_r[i,:,:,:], (n,o)), cmap='cmo.ice_r', extent=extent, aspect='auto', origin='lower')
        plt.xlabel('Time (s)')
        if i == 0:
            plt.ylabel('Frequency (Hz)')
        if counter == 0:
            plt.figtext(0.03, 0.87, 'a)', size=16, fontweight='bold')
            plt.figtext(0.03, 0.5, 'b)', size=16, fontweight='bold')
            plt.figtext(0.03, 0.42, 'c)', size=16, fontweight='bold')
        counter += 1

    fig.suptitle(figtitle, size=16, weight='bold')
    # fig.tight_layout()
    fig.subplots_adjust(top=0.88, left=0.08)
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def view_specgram_training2(
        x, x_r, z, n, o,
        figtitle,
        disp_idx,
        tvec,
        fvec,
        fname_dataset,
        figsize=(8,3),
        show=True
    ):
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True, # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}"""],
    }
    mpl.rcParams.update(rc_fonts)
    sample_idx = np.arange(0, len(disp_idx))
    metadata = get_metadata(sample_idx, disp_idx, fname_dataset)
    X = x.detach().cpu().numpy()
    X_r = x_r.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    fig = plt.figure(figsize=(figsize[0],len(disp_idx)*figsize[1]), dpi=100)
    heights = [4 for i in range(x.size()[0])]
    widths = [3, 0.5, 3]
    extent = [min(tvec), max(tvec), min(fvec), max(fvec)]
    arrow_yloc = 1.12
    gs = gridspec.GridSpec(nrows=x.size()[0], ncols=3, height_ratios=heights, width_ratios=widths, wspace=0.2, hspace=0.6)
    counter = 0
    for i in range(x.size()[0]):
        station = metadata[i]['Station']
        try:
            time_on = datetime.strptime(metadata[i]['TriggerOnTime'],
                                        '%Y-%m-%dT%H:%M:%S.%f').strftime(
                                        '%Y-%m-%dT%H:%M:%S.%f')[:-4]
        except:
            time_on = datetime.strptime(metadata[i]['TriggerOnTime'],
                                        '%Y-%m-%dT%H:%M:%S').strftime(
                                        '%Y-%m-%dT%H:%M:%S.%f')[:-4]
        ax1 = fig.add_subplot(gs[counter,0])
        plt.imshow(np.reshape(X[i,:,:,:], (n,o)), cmap='cmo.ice_r', extent=extent, aspect='auto', origin='lower')
        plt.colorbar(orientation='vertical', pad=0)
        plt.clim(0,1)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Input\n' + r'$\bm{x}$', x=0.55)

        xy1 = (0.67, arrow_yloc)

        ax2 = fig.add_subplot(gs[counter,1])
        plt.imshow(np.expand_dims(z[i], 1), cmap=cmo.deep_r, aspect='auto')
        for j in range(z.shape[1]):
            plt.text(
                0,
                j,
                f"{z[i,j]:.1f}",
                backgroundcolor='w',
                ha='center',
                va='center',
                bbox=dict(boxstyle='square,pad=0', facecolor='w', edgecolor='w')
            )
        plt.xticks([])
        plt.yticks(ticks=np.arange(0, z.shape[1]), labels=np.arange(1, z.shape[1]+1))
        plt.title('Latent Space\n' + r'$\bm{z}$')

        xy2 = (-0.8, arrow_yloc)
        con = ConnectionPatch(
            xyA=xy1,
            xyB=xy2,
            coordsA="axes fraction",
            coordsB="axes fraction",
            axesA=ax1,
            axesB=ax2,
            arrowstyle="simple",
            color="k"
        )
        ax2.add_artist(con)

        ax3 = fig.add_subplot(gs[counter,2])
        plt.imshow(np.reshape(X_r[i,:,:,:], (n,o)), cmap='cmo.ice_r', extent=extent, aspect='auto', origin='lower')
        plt.colorbar(orientation='vertical', pad=0)
        plt.clim(0,1)
        plt.title("Output\n" + r"$\bm{x}'$", x=0.55)

        xy2 = (1.8, arrow_yloc)
        xy3 = (0.4, arrow_yloc)
        con = ConnectionPatch(
            xyA=xy2,
            xyB=xy3,
            coordsA="axes fraction",
            coordsB="axes fraction",
            axesA=ax2,
            axesB=ax3,
            arrowstyle="simple",
            color="k"
        )
        ax3.add_artist(con)

        counter += 1
    fig.subplots_adjust(top=0.86, bottom=0.15)
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def view_TSNE(results, labels, title, show=False):
    label_list, counts = np.unique(labels, return_counts=True)

    colors = cmap_lifeaquatic(len(counts))
    data = np.stack([(labels+1), results[:,0], results[:,1]], axis=1)
    df = pd.DataFrame(data=data, columns=["Label", "x", "y"])
    df["Label"] = df["Label"].astype('int').astype('category')

    fig = plt.figure(figsize=(6,8))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])

    ax = fig.add_subplot(gs[0])
    sns.scatterplot(data=df, x="x", y="y", hue="Label", palette=colors, alpha=0.2)
    plt.axis('off')
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.75), ncol=1)
    plt.title(title)

    ax = fig.add_subplot(gs[1])
    arr = plt.hist(labels+1, bins=np.arange(1, max(labels)+3, 1), histtype='bar', align='left', rwidth=0.8, color='k')
    plt.grid(axis='y', linestyle='--')
    plt.ylim([0, 1.25 * max(counts)])
    plt.xlabel('Cluster Label')
    plt.ylabel('Detections')
    plt.title(f'Cluster Assignments of Detections, N = {len(labels)}')

    N = counts.sum()
    def CtP(x):
        return 100 * x / N

    def PtC(x):
        return x * N / 100

    ax2 = ax.secondary_yaxis('right', functions=(CtP, PtC))
    ax2.set_ylabel('% of N')
    for i in range(len(np.unique(labels))):
        plt.text(arr[1][i], 1.05 * arr[0][i], str(int(arr[0][i])), ha='center')

    if show:
        plt.show()
    else:
        plt.close()
    return fig
