#!/usr/bin/env python3

import os

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
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from RISCluster import utils
from RISCluster.networks import AEC, DEC
from RISCluster.environment import EnvironmentCatalogue


def analyze_dec(
        model,
        dataloader,
        device,
        fname_dataset,
        index_tra,
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        tsne_results,
        epoch,
        show=False,
        latex=False
    ):
    '''
    Function displays reconstructions using the centroids of the latent feature
    space.
    # Arguments
        model: PyTorch model instance
        dataloader: PyTorch dataloader instance
        labels: Vector of cluster assignments
        device: PyTorch device object ('cpu' or 'gpu')
        epoch: Training epoch, used for title.
    # Input:
        2D array of shape [n_samples, n_features]
    # Output:
        Figures displaying centroids and their associated reconstructions.
    '''
    n_clusters = model.n_clusters
    p = 2
    title = f't-SNE Results - Epoch {epoch}'
    if tsne_results is not None:
        fig1 = view_TSNE(tsne_results, labels_b, title, show)
    else:
        raise("TSNE within function not implemented.")
        # fig1 = view_TSNE(tsne(data_b), labels_b, title, show)
    fig2 = cluster_gallery(
        model,
        dataloader.dataset,
        fname_dataset,
        index_tra,
        device,
        data_b,
        labels_b,
        centroids_b,
        p,
        show,
        latex
    )
    # fig3 = centroid_dashboard(
    #     data_b,
    #     labels_b,
    #     centroids_b,
    #     n_clusters,
    #     p,
    #     show
    # )
    fig3 = centroid_distances(
        data_b,
        labels_b,
        centroids_b,
        n_clusters,
        p,
        show
    )
    fig4 = view_latent_space(
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        n_clusters,
        p,
        show,
        latex
    )
    fig5 = view_class_cdf(
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        n_clusters,
        p,
        show,
        latex
    )
    fig6 = view_class_pdf(
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        n_clusters,
        p,
        show,
        latex
    )
    return [fig1, fig2, fig3, fig4, fig5, fig6]


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
        (56, 2, 130),
        (0, 0, 0)
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
        plt.imshow(centroids[l][None].T, cmap=cmap, vmax=vmax, interpolation="none")
        plt.xticks([])
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'))
        plt.ylabel('Centroid Feature')
        # Dataset Latent Features
        ax = fig.add_subplot(gs_sub[0,1])
        plt.imshow(z_array[sort_index_d].T, cmap=cmap, aspect='auto', vmax=vmax, interpolation="none")
        plt.vlines(centroids_ind, -0.5, d-0.5, colors='w', linestyles='dotted')
        for ll in range(n_clusters-1):
            plt.text(centroids_ind[ll], ll+1, str(labels_not[ll]+1), backgroundcolor='w', ha='center', bbox=dict(boxstyle='square,pad=0', facecolor='w', alpha=0.5, edgecolor='w'))
        plt.xticks([])
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'))
        ax.yaxis.tick_right()
        plt.ylabel('Latent Feature')
        ax.yaxis.set_label_position('right')
        plt.title(f"Class {l+1}: Dataset")
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
        plt.imshow(centroids[l][None].T, cmap=cmap, vmax=vmax, interpolation="none")
        plt.xticks([])
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'))
        plt.ylabel('Centroid Feature')
        # Cluster Latent Features
        ax = fig.add_subplot(gs_sub[0,1])
        tmp = z_sub.T
        plt.imshow(np.concatenate((tmp, np.zeros((tmp.shape[0], counts.max() - tmp.shape[1]))), axis=1), cmap=cmap, extent=extent, aspect='auto', vmax=vmax, interpolation="none")
        plt.xticks([])
        # plt.yticks([])
        # To-do: Fix yticks
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'))
        ax.yaxis.tick_right()
        plt.ylabel('Latent Feature')
        ax.yaxis.set_label_position('right')
        plt.title(f"Class {l+1}: Within-class")
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
    plt.imshow(dist_mat, cmap='cmo.solar_r', origin='lower', interpolation="none")
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
        show=True,
        latex=False
    ):
    model.eval()
    label_list, counts = np.unique(labels, return_counts=True)
    if centroids is not None:
        X_c = model.decoder(torch.from_numpy(centroids).float().to(device))
    else:
        centroids = model.clustering.weights
        X_c = model.decoder(centroids)
        centroids = centroids.detach().cpu().numpy()
    N = 6
    if latex:
        params = {
            'text.usetex': True,
            'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amsbsy}']
        }
        plt.rcParams.update(params)
    fig = plt.figure(figsize=(len(label_list),2*N), dpi=150)
    heights = [1 for i in range(N+1)] + [0.2]
    gs_sup = gridspec.GridSpec(nrows=N+2, ncols=len(label_list), hspace=0.1, wspace=0.1, height_ratios=heights)
    heights = [1, 4, 0.5]
    font = {
        'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 5,
    }
    fontsize = 16
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
            idx, batch = batch
            idx.numpy()
            X = batch.to(device)
            with h5py.File(fname_dataset, 'r') as f:
                M = len(idx)
                DataSpec = '/4.0/Trace'
                dset = f[DataSpec]
                k = 199

                tr = np.empty([M, k])
                dset_arr = np.empty([k,])

                for i in range(M):
                    dset_arr = dset[idx[i]]
                    tr[i,:] = dset_arr

        Z = model.encoder(X)

        gs_sub = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_sup[0,l], hspace=0, wspace=0, height_ratios=heights)
        ax = fig.add_subplot(gs_sub[0])
        plt.axis('off')
        ax = fig.add_subplot(gs_sub[1])
        plt.imshow(torch.squeeze(X_c[l]).detach().cpu().numpy(), cmap=cmap_spec, aspect='auto', origin='lower', interpolation="none")
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(f"$j={label+1}$", va="bottom", size=fontsize)
        if l == 0:
            if latex:
                plt.ylabel(r"$g_\theta(\pmb{\mu}_j)$", rotation=0, va="center", ha="right", size=fontsize)
            else:
                plt.ylabel("$g(mu)", rotation=0, va="center", ha="right")

        ax = fig.add_subplot(gs_sub[2])
        plt.imshow(np.expand_dims(centroids[l], 0), cmap=cmap_feat, aspect='auto', vmax = vmax, interpolation="none")
        plt.xticks([])
        plt.yticks([])
        if l == 0:
            if latex:
                plt.ylabel(r'$\pmb{\mu}_j$', rotation=0, va="center", ha="right", size=fontsize)
            else:
                plt.ylabel("mu", rotation=0, va="center", ha="right")

        for i in range(N):
            gs_sub = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_sup[i+1,l], hspace=0, wspace=0, height_ratios=heights)
            ax = fig.add_subplot(gs_sub[0])
            plt.plot(tr[i], 'k', linewidth=0.5)
            plt.xlim((0, k))
            plt.xticks([])
            plt.yticks([])

            ax = fig.add_subplot(gs_sub[1])
            plt.imshow(np.squeeze(X[i,:,:].detach().cpu().numpy()), cmap=cmap_spec, aspect='auto', origin='lower', interpolation="none")
            # plt.text(0, 60, f"{load_index[i]}", fontdict=font)
            # plt.text(110, 60, f"d={distance[i]:.1f}", fontdict=font)
            if (l == 0) and (i == N - 1):
                pass
            else:
                plt.xticks([])
                plt.yticks([])

            if l == 0:
                if latex:
                    plt.ylabel(fr"$\pmb{{x}}_{i+1}$", rotation=0, va="center", ha="right", size=fontsize)
                else:
                    plt.ylabel(f"x_{i+1}", rotation=0, va="center", ha="right")

            ax = fig.add_subplot(gs_sub[2])
            plt.imshow(np.expand_dims(Z[i].detach().cpu().numpy(), 0), cmap=cmap_feat, aspect='auto', vmax = vmax, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            if l == 0:
                if latex:
                    plt.ylabel(fr"$\pmb{{z}}_{i+1}$", rotation=0, va="center", ha="right", size=fontsize)
                else:
                    plt.ylabel(f"z_{i+1}", rotation=0, va="center", ha="right")

    gs_sub = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_sup[-1,:])
    # Colorbar: Specgram
    ax = fig.add_subplot(gs_sub[0])
    plt.axis('off')
    cmap = 'cmo.deep_r'
    axins = inset_axes(ax, width="75%", height="50%", loc="center")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_spec), cax=axins, orientation='horizontal')
    cbar.set_label('Normalized Spectrogram Value')
    cbar.ax.tick_params(labelsize=10)

    # Colorbar: Latent Space
    ax = fig.add_subplot(gs_sub[1])
    plt.axis('off')
    cmap = 'cmo.deep_r'
    axins = inset_axes(ax, width="75%", height="50%", loc="center")
    norm = mpl.colors.Normalize(vmin=z_array.min(), vmax=vmax)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_feat), cax=axins, orientation='horizontal')
    cbar.set_label('Latent Feature Value')
    cbar.ax.tick_params(labelsize=10)

    if show:
        plt.show()
    else:
        plt.close()
    return fig


def compare_images(
        model,
        epoch,
        image_index,
        fname_dataset,
        device,
        savepath=None,
        show=True,
    ):
    tvec, fvec = utils.get_timefreqvec(fname_dataset)
    dataset = utils.H5SeismicDataset(
        fname_dataset,
        transform = transforms.Compose(
            [utils.SpecgramShaper(), utils.SpecgramToTensor()]
        )

    )
    subset = Subset(dataset, image_index)
    dataloader = DataLoader(subset, batch_size=len(image_index))
    model.eval()

    for batch in dataloader:
        idx, X = batch
        Xr, z = model(X.to(device))

    figtitle = f'Pre-training: Epoch {epoch}'
    if savepath is not None:
        savepath_snap = savepath + '/snapshots/'
        if not os.path.exists(savepath_snap):
            os.makedirs(savepath_snap)

    fig = view_specgram_training(
        X,
        Xr, z,
        figtitle,
        image_index,
        tvec,
        fvec,
        fname_dataset,
        show=show
    )
    if savepath is not None:
        figname = savepath_snap + f'AEC_Training_Epoch_Single_{epoch:03d}.png'
        fig.savefig(figname, dpi=300)
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


def plotter_mp(
        fignames,
        figpaths,
        tb,
        model,
        dataloader,
        device,
        fname_dataset,
        index_tra,
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        tsne_results,
        epoch,
        show,
        latex=False
    ):
    '''Wrapper function for plotting DEC training and performance.

    Parameters
    ----------
    fignames : list
        List of figure names

    figpaths : list
        List of paths where to save figures

    tb : Tensorboard SummaryWriter object

    model : PyTorch model instance

    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    device : PyTorch device object ('cpu' or 'gpu')

    fname_dataset : str
        Path to dataset

    index_tra: array
        Indeces of data samples to be used for DEC training.

    data_a : array (M,D)
        Latent data from model initialization [m_samples,d_features]

    data_b : array (M,D)
        Latent data from current model state [m_samples,d_features]

    labels_a : array (M,)
        Class labels from cluster initialization

    labels_b : array (M,)
        Class labels from current state of clustering

    centroids_a : array (n_clusters,)
        Cluster centroids from model initialization

    centroids_b : array (n_clusters,)
        Current cluster centroids

    tsne_results : array (M,2)
        2-D t-SNE results from current model output

    epoch : int
        Current epoch of training

    show : boolean
        Show figures or not

    latex : boolean (default=False)
        Compile figures using latex (extremely slow - not recommended unless
        rendering figures for publishing)

    Outputs to Disk
    ---------------
    Figures analyzing DEC performance.
    '''

    figures = analyze_dec(
        model,
        dataloader,
        device,
        fname_dataset,
        index_tra,
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        tsne_results,
        epoch,
        show,
        latex
    )
    [fig.savefig(f"{figpaths[i]}/{fignames[i]}_{epoch:03d}.png", dpi=300) \
        for i, fig in enumerate(figures)]
    [tb.add_figure(f"{fignames[i]}", fig, global_step=epoch, close=True) \
        for i, fig in enumerate(figures)]


def save_DEC_output(x, label, x_rec, z, idx, savepath):
    # print(f'x={type(x)} | label={type(label)} | x_r={type(x_rec)} | z={type(z)} | idx={type(idx)} | path={type(savepath)}')
    fig = view_DEC_output(x, label, x_rec, z, idx, show=False)
    # print(f'{savepath}{idx:07d}.png')
    fig.savefig(f'{savepath}/{idx:07d}.png', dpi=300)
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
        plt.imshow(torch.reshape(X_r[i,:,:,:], (n,o)).cpu().detach().numpy(), aspect='auto', interpolation="none")
        plt.gca().invert_yaxis()
        plt.title(f'Class {i}')

        ax = fig.add_subplot(gs[i,1])
        plt.imshow(np.expand_dims(centroids[i].cpu().detach().numpy(), 1), cmap='viridis', aspect='auto', interpolation="none")
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
        show=True,
        latex=False
    ):
    def _roundup(x):
        return int(np.ceil(x / 5.0)) * 5

    label_list, counts_a = np.unique(labels_a, return_counts=True)
    _, counts_b = np.unique(labels_b, return_counts=True)

    fig = plt.figure(figsize=(7, 2*int(np.ceil(n_clusters/2))), dpi=150)
    fontsize = 16
    if latex:
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    colors = cmap_lifeaquatic(n_clusters)
    gs = gridspec.GridSpec(nrows=int(np.ceil(n_clusters/2)), ncols=2, hspace=0, wspace=0)
    max_dist = 0
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

        max_dist_ = np.max([distance_a.max(), distance_b.max()])
        # max_dist_ = _roundup(max_dist_)
        if max_dist_ > max_dist:
            max_dist = max_dist_

        plt.plot(distance_a, cdf_a, color=colors[0], label="GMM")
        plt.plot(distance_b, cdf_b, color=colors[1], label="DEC")
        ax.set_yticks([0., 0.5, 1.])

        if ((n_clusters % 2 == 0) and (l == n_clusters - 2)) or ((n_clusters % 2 != 0) and (l == n_clusters - 1)):
            if latex:
                plt.xlabel(fr"$d=\Vert\pmb{{z}} - \pmb{{\mu}}_j\Vert_{p}$", size=fontsize)
            else:
                plt.xlabel(f"d_{p}", size=fontsize)
            plt.ylabel("CDF", size=fontsize)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    allaxes = fig.get_axes()
    for j, ax in enumerate(allaxes):
        ax.set_xlim(0, max_dist)
        ax.text(0.9*max_dist, 0.15, f"$j={j+1}$", ha="right", va="bottom", fontsize=fontsize)

    handles, labels = ax.get_legend_handles_labels()
    if len(label_list) % 2 != 0:
        fig.legend(handles, labels, loc=(0.65, 0.1), fontsize=fontsize)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=fontsize)
        plt.subplots_adjust(bottom=0.15)

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
        show=True,
        latex=False
    ):
    def _roundup(x):
        return int(np.ceil(x / 10.0)) * 10

    label_list, counts_a = np.unique(labels_a, return_counts=True)
    _, counts_b = np.unique(labels_b, return_counts=True)
    dx = 1
    nbins = 200
    X = np.linspace(0, 201, nbins)

    fig = plt.figure(figsize=(12, 2.5*int(np.ceil(n_clusters/2))), dpi=150)
    fontsize = 20
    if latex:
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    gs = gridspec.GridSpec(nrows=int(np.ceil(n_clusters/2)), ncols=2, hspace=0.3, wspace=0.05)
    colors = cmap_lifeaquatic(n_clusters)
    max_dist = 0
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

        max_dist_ = np.max([distance_a.max(), distance_b.max()])
        max_dist_ = _roundup(max_dist_)
        if max_dist_ > max_dist:
            max_dist = max_dist_

        axa = fig.add_subplot(gs_sub[0])
        for ll in range(n_clusters):
            query_a = np.where(labels_a_ == label_list[ll])[0]
            distance_a_ = distance_a[query_a]
            hist_a = np.histogram(distance_a_, bins=X, density=True)[0]

            plt.plot(X[:-1], hist_a, color=colors[ll], label=f"{ll+1}")
            plt.fill_between(X[:-1], 0, hist_a, color=colors[ll], alpha=0.2)
            plt.xlim(X.min(), X.max())
            plt.xticks([])
            plt.ylim(0, 1)
            plt.yticks([0., 0.5, 1.])
            plt.text(1, 0.9, 'GMM', ha='right', va='top', transform=axa.transAxes, fontsize=fontsize)
            if latex:
                plt.title(fr"Class PDFs relative to $\pmb{{\mu}}_{l+1}$", loc="left", size=fontsize)
            else:
                plt.title(f"Class PDFs relative to mu_{l+1}", loc="left", size=fontsize)

        axb = fig.add_subplot(gs_sub[1])
        for ll in range(n_clusters):
            query_b = np.where(labels_b_ == label_list[ll])[0]
            distance_b_ = distance_b[query_b]
            hist_b = np.histogram(distance_b_, bins=X, density=True)[0]

            plt.plot(X[:-1], hist_b, color=colors[ll], label=f"{ll+1}")
            plt.fill_between(X[:-1], 0, hist_b, color=colors[ll], alpha=0.2)
            plt.xlim(X.min(), X.max())
            plt.ylim(0, 1)
            plt.yticks([0., 0.5, 1.])
            plt.text(1, 0.9, 'DEC', ha='right', va='top', transform=axb.transAxes, fontsize=fontsize)

        if ((n_clusters % 2 == 0) and (l == n_clusters - 2)) or ((n_clusters % 2 != 0) and (l == n_clusters - 1)):
            if latex:
                plt.xlabel(fr"$d=\Vert\pmb{{z}} - \pmb{{\mu}}_j\Vert_{p}$", size=fontsize)
            else:
                plt.xlabel(f"d_{p}", size=fontsize)
            plt.ylabel("PDF", size=fontsize)
            axa.set_xticklabels([])
            axa.set_yticklabels([])
        else:
            axa.set_xticklabels([])
            axa.set_yticklabels([])
            axb.set_xticklabels([])
            axb.set_yticklabels([])

    allaxes = fig.get_axes()
    for ax in allaxes:
        ax.set_xlim(0, max_dist)

    handles, labels = ax.get_legend_handles_labels()
    if len(label_list) % 2 != 0:
        leg = fig.legend(handles, labels, loc=(0.54, 0.1), ncol=int(np.ceil(n_clusters/2)), fontsize=fontsize-2)
    else:
        leg = fig.legend(handles, labels, loc=(0.13, 0.005), ncol=n_clusters, fontsize=fontsize-4)
        plt.subplots_adjust(bottom=0.15)
    leg.set_title("Classes", prop={'size':'xx-large'})

    if show:
        plt.show()
    else:
        plt.close()
    return fig


def view_cluster_stats(k_list, inertia, silh, gap_g, gap_u, show=False):
    def _make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, host = plt.subplots(figsize=(6,4), dpi=150)

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


def view_DEC_output(x, label, x_rec, z, idx, figsize=(12,9), show=False):
    fig = plt.figure(figsize=figsize, dpi=150)
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1,0.1,1])
    # Original Spectrogram
    ax = fig.add_subplot(gs[0])
    plt.imshow(np.squeeze(x), aspect='auto', interpolation="none")
    plt.ylabel('Frequency Bin')
    plt.xlabel('Time Bin')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Original Spectrogram')
    # Latent Space Representation:
    ax = fig.add_subplot(gs[1])
    plt.imshow(np.expand_dims(z, 1), cmap='viridis', aspect='auto', interpolation="none")
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
    plt.imshow(np.squeeze(x_rec), aspect='auto', interpolation="none")
    plt.ylabel('Frequency Bin')
    plt.xlabel('Time Bin')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Reconstructed Spectrogram')
    fig.suptitle(f'Class: {label}', size=18, weight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if show is False:
        plt.close()
    else:
        plt.show()
    return fig


def view_detections(fname_dataset, image_index, figsize=(12,9), show=True):
    '''Plots selected spectrograms & traces.'''
    sample_index = np.arange(0, len(image_index))
    dataset = utils.H5SeismicDataset(
        fname_dataset,
        transform = transforms.Compose(
            [utils.SpecgramShaper(), utils.SpecgramToTensor()]
        )
    )
    subset = Subset(dataset, image_index)
    dataloader = DataLoader(subset, batch_size=len(image_index))

    for batch in dataloader:
        idx, X = batch
        idx.numpy()
        with h5py.File(fname_dataset, 'r') as f:
            M = len(idx)
            DataSpec = '/4.0/Trace'
            dset = f[DataSpec]
            k = 199

            tr = np.empty([M, k])
            dset_arr = np.empty([k,])

            for i in range(M):
                dset_arr = dset[idx[i]]
                tr[i,:] = dset_arr

    factor = 1e-9
    tr_max = np.max(np.abs(tr)) / factor

    tvec, fvec = utils.get_timefreqvec(fname_dataset)

    extent = [min(tvec), max(tvec), min(fvec), max(fvec)]
    metadata = utils.get_metadata(sample_index, image_index, fname_dataset)
    fontsize = 16
    fig = plt.figure(figsize=figsize, dpi=150)
    cmap = 'cmo.ice_r'
    gs_sup = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.35, wspace=0.15)
    counter = 0
    for i in range(len(sample_index)):
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_sup[i], hspace=0)

        ax = fig.add_subplot(gs_sub[0])
        plt.imshow(X[sample_index[i],:,:].squeeze(), extent=extent, aspect='auto', origin='lower', cmap=cmap, interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([5, 10, 15, 20])
        if i == 0:
            plt.ylabel('Frequency\n(Hz)', size=fontsize)
        station = metadata[counter]['station']
        time_on = metadata[counter]['spec_start']
        plt.title(f'Station {station}, {time_on[:-4]}', size=fontsize)
                  # f'Index: {image_index[sample_index[i]]}')

        tvec = np.linspace(extent[0], extent[1], tr.shape[1])

        ax = fig.add_subplot(gs_sub[1])
        plt.plot(tvec, tr[i,:] / factor)
        plt.xlim(min(tvec), max(tvec))
        plt.ylim(-tr_max, tr_max)
        ax.set_xticks(np.arange(5))
        if i == 0:
            plt.xlabel('Time (s)', size=fontsize)
            plt.ylabel('Acceleration\n(nm/s$^2$)', size=fontsize)

        counter += 1

    if show:
        plt.show()
    else:
        plt.close()
    return fig


def view_history_AEC(path, show=False):
    df = pd.read_csv(path, index_col=0)

    fig = plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(df.index, df["Training Loss"], color='k', linestyle='solid')
    plt.plot(df.index, df["Validation Loss"], color='k', linestyle='dashdot')
    plt.xlim(0, max(df.index))
    plt.yscale('log')
    plt.grid(which="both")
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend(['Training Loss','Validation Loss'])
    plt.title('Autoencoder Training')

    if show:
        plt.show()
    else:
        plt.close()
    return fig


def view_history_DEC(paths, show=False):
    df1 = pd.read_csv(paths[0], index_col=0)
    df2 = pd.read_csv(paths[1], index_col=0)

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax1.plot(df1.index, df1["Total Loss"], color='k', linestyle='solid')
    ax1.plot(df1.index, df1["Clustering Loss"], color='r', linestyle='dashdot')
    ax1.plot(df1.index, df1["Reconstruction Loss"], color='b', linestyle='dashed')
    ax1.set_xlim(0, max(df1.index))
    ax1.set_yscale('log')
    ax1.grid(which="both")
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis='x', length=0)
    ax1.set_ylabel('Loss')
    ax1.set_title('DEC Model Training')
    ax1.legend(['Total','$L_{KLD}$','$L_{MSE}$'])

    ax2.plot(df2.index, df2["Delta"], color='k')
    ax2.set_yscale('log')
    ax2.grid(which="both")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel(r'$\Delta$ Assignments (\%)')

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
        show=True,
        latex=False
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
    if latex:
        params = {
            'text.usetex': True,
            'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amsbsy}']
        }
        plt.rcParams.update(params)
    fontsize = 18
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
        ax0 = fig.add_subplot(gs_sub[0,0])
        plt.imshow(centroids_a[l][None].T, cmap=cmap, vmax=vmax, interpolation="none")
        plt.xticks([])
        if l == 0:
            plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'), size=5)
            plt.ylabel('GMM', size=fontsize)
            if latex:
                plt.title(r'$\pmb{\mu}_j$')
            else:
                plt.title('mu_j')
        else:
            plt.yticks(ticks=np.linspace(0,d-1,d), labels=[], size=5)

        # Latent Space A
        ax1 = fig.add_subplot(gs_sub[0,1])
        plt.imshow(data_a[sort_index_d].T, cmap=cmap, aspect='auto', vmax=vmax, interpolation="none")
        plt.vlines(centroids_ind, -0.5, d-0.5, colors='w', ls='dashed', lw=0.75, alpha=0.5)
        for ll in range(n_clusters-1):
            if latex:
                plt.text(centroids_ind[ll], 1.1*(ll+1), f"$\pmb{{\mu}}_{labels_not[ll]+1}$", size=6, backgroundcolor='w', ha='center', bbox=dict(boxstyle='square,pad=0', facecolor='w', alpha=1, edgecolor='w'))
            else:
                plt.text(centroids_ind[ll], 1.1*(ll+1), f"u{labels_not[ll]+1}", size=6, backgroundcolor='w', ha='center', bbox=dict(boxstyle='square,pad=0', facecolor='w', alpha=1, edgecolor='w'))
        plt.xticks([])
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=[])
        if l == 0:
            if latex:
                plt.text(0.03, 1.1, f"$\pmb{{z}}_i \in Z$", size=fontsize, transform=ax1.transAxes)
            else:
                plt.text(0.03, 1.1, f"z", size=fontsize, transform=ax1.transAxes)
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
        ax2 = fig.add_subplot(gs_sub[1,0])
        plt.imshow(centroids_b[l][None].T, cmap=cmap, vmax=vmax, interpolation="none")
        plt.xticks([])
        if l == 0:
            plt.yticks(ticks=np.linspace(0,d-1,d), labels=np.linspace(1,d,d, dtype='int'), size=5)
            plt.ylabel('DEC', size=fontsize)
        else:
            plt.yticks(ticks=np.linspace(0,d-1,d), labels=[], size=5)

        # Latent Space B
        ax3 = fig.add_subplot(gs_sub[1,1])
        plt.imshow(data_b[sort_index_d].T, cmap=cmap, aspect='auto', vmax=vmax, interpolation="none")
        plt.vlines(centroids_ind, -0.5, d-0.5, colors='w', ls='dashed', lw=0.75, alpha=0.5)
        for ll in range(n_clusters-1):
            if latex:
                plt.text(centroids_ind[ll], 1.1*(ll+1), f"$\pmb{{\mu}}_{labels_not[ll]+1}$", size=6, backgroundcolor='w', ha='center', bbox=dict(boxstyle='square,pad=0', facecolor='w', alpha=1, edgecolor='w'))
            else:
                plt.text(centroids_ind[ll], 1.1*(ll+1), f"u{labels_not[ll]+1}", size=6, backgroundcolor='w', ha='center', bbox=dict(boxstyle='square,pad=0', facecolor='w', alpha=1, edgecolor='w'))
        if l == 0:
            label = ax3.set_xlabel("$i$", size=14)
            ax3.xaxis.set_label_coords(-0.03, 0)
        else:
            xlabels = [item.get_text() for item in ax3.get_xticklabels()]
            empty_string_labels = ['']*len(xlabels)
            ax3.set_xticklabels(empty_string_labels)
        plt.yticks(ticks=np.linspace(0,d-1,d), labels=[])

    # Colorbar
    ax4 = fig.add_axes([0, 0.045, 1, 0.1])
    plt.axis('off')
    axins = inset_axes(ax4, width="50%", height="15%", loc="center")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axins, orientation='horizontal')
    cbar.set_label('Latent Feature Value', size=fontsize)
    if latex:
        fig.suptitle(f"Latent space sorted by $d_{{i,j}}=\Vert \pmb{{z}}_i-\pmb{{\mu}}_j \Vert_{p} \mid d_{{i+1,j}} > d_{{i,j}}$", size=18)
    else:
        fig.suptitle(f"Latent space sorted by d_{p}", size=18)
    fig.subplots_adjust(top=0.91)

    if show:
        plt.show()
    else:
        plt.close()
    return fig


def view_learningcurve(training_history, validation_history, show=True):
    epochs = len(training_history['mse'])
    fig = plt.figure(figsize=(18,6), dpi=150)
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
    fig = plt.figure(figsize=(6,6), dpi=150)
    sns.scatterplot(pca2d[:,0], pca2d[:,1], hue=labels, palette='Set1', alpha=0.2)
    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    fig.tight_layout()
    return fig


def view_series(
        station,
        aws,
        path_to_data,
        path_to_catalogue,
        path_to_labels,
        env_vars=["tide","sea_ice_conc","temp","wind_spd"],
        times=None,
        vlines=None,
        freq="month",
        label_list=[i for i in range(8)],
        log=False,
        maxcounts=False,
        figsize=(12,9),
        title=None,
        showlabels=True,
        show=False
    ):
    df_env = EnvironmentCatalogue(station, aws, path_to_data).df
    catalogue = utils.LabelCatalogue([path_to_catalogue, path_to_labels])
    counts = catalogue.gather_counts(station=station, freq=freq, label_list=label_list)
    if times is not None:
        mask = (df_env.index >= times[0]) & (df_env.index <= times[1])
        df_env = df_env.loc[mask]
        mask = (counts.index >= times[0]) & (counts.index <= times[1])
        counts = counts.loc[mask]
    # =======================================
    fig = plt.figure(figsize=figsize)

    fontsize = 16
    if times is not None:
        duration = times[1] - times[0]
        if duration.days >= 28:
            xticks = pd.date_range(df_env.index[0], df_env.index[-1], freq="7D")
            xlabels = xticks.strftime("%d %b\n%Y")
            xlabelrotation = 0
            xlabelalignment = "center"
        elif duration.days > 7:
            xticks = pd.date_range(df_env.index[0], df_env.index[-1], freq="D")
            xlabels = xticks.strftime("%Y-%b-%d")
            xlabelrotation = 45
            xlabelalignment = "right"
        else:
            xticks = pd.date_range(df_env.index[0], df_env.index[-1], freq="D")
            xlabels = xticks.strftime("%Y-%b-%d")
            xlabelrotation = 45
            xlabelalignment = "right"
    else:
        xticks = pd.date_range(df_env.index[0], df_env.index[-1], freq="3MS")
        xlabels = xticks.strftime("%b\n%Y")
        xlabelrotation = 0
        xlabelalignment = "center"

    linewidth = 0.75
    colors = cmap_lifeaquatic()

    env_keys = df_env.keys().tolist()
    labels = counts.keys().sort_values()
    if not maxcounts:
        maxcounts = counts.max().max()
    heights = [1 for i in range(len(env_vars))] + [0.5 for i in range(len(label_list))]

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    gs = fig.add_gridspec(len(env_vars) + len(labels), 1, height_ratios=heights, hspace=0.4)

    params = dict(
        env_key=env_keys,
        ylabels=[
            "Tide (m)",
            "% Sea Ice\n    Coverage",
            "Temperature\n          ($^\degree$C)",
            "Wind Speed\n         (m/s)",
            "Net Surf.\n Melt Energy",
            "Wave\nAmplitude"],
        ylims=[(-1, 1), (0, 100), (-60, 10), (0, 25), (-50, 50), (0, 0.008)],
        colors=[colors[i] for i in [11, 10, 0, 3, 1, 2]]
    )
    sel_ind = [env_keys.index(key) for key in env_vars]
    params = dict(
        env_key=[params["env_key"][i] for i in sel_ind],
        ylabels=[params["ylabels"][i] for i in sel_ind],
        ylims=[params["ylims"][i] for i in sel_ind],
        colors=[params["colors"][i] for i in sel_ind]
    )

    for i, key in enumerate(env_vars):
        ax = fig.add_subplot(gs[i])
        plt.plot(df_env[key], color=params["colors"][i], linewidth=linewidth)
        if vlines is not None:
            [plt.axvline(pd.Timestamp(dt), color="r") for dt in vlines]
        plt.grid(axis="x")
        ax.set_xlim(df_env.index[0], df_env.index[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
        ax.spines["top"].set_visible(False)
        ax.set_ylim(params["ylims"][i])
        plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
        if showlabels:
            plt.ylabel(f"{alphabet[i]}) {params['ylabels'][i]}", rotation=0, ha="left", va="center", fontsize=fontsize)
            ax.yaxis.set_label_coords(-0.22, 0.5)
        if i == 0 and title is not None:
            plt.title(title, fontsize=fontsize+2)

    for j, key in enumerate(labels):
        ax = fig.add_subplot(gs[i + j + 1])
        plt.plot(counts[key], color="navy")
        if vlines is not None:
            [plt.axvline(pd.Timestamp(dt), color="r") for dt in vlines]
        ax.grid(axis="x")
        ax.set_xlim(df_env.index[0], df_env.index[-1])
        if j == len(labels)-1:
            ax.set_xticklabels(xlabels, rotation=xlabelrotation, ha=xlabelalignment, size=fontsize)
            ax.spines["bottom"].set_visible(True)
            ax.set_xticks(xticks)
        else:
            ax.set_xticklabels([])
            ax.set_xticks(xticks)
        ax.spines["top"].set_visible(False)
        if log:
            plt.yscale("log")
            ax.set_ylim(1, 100)
            ax.set_yticks([1, 10, 100])
        else:
            ymax = maxcounts
            ax.set_ylim(0, ymax)
            ax.set_yticks([0, ymax/2, ymax])
        if j == 0:
            if log:
                ax.set_yticklabels([1, 10, 100], size=fontsize-2)
            else:
                ax.set_yticklabels([0, int(ymax/2), ymax], size=fontsize-2)
        else:
            ax.set_yticklabels([])
        if showlabels:
            plt.ylabel(f"{alphabet[i + j + 1]}) Class {j + 1}", rotation=0, ha="left", va="center", fontsize=fontsize)
            ax.yaxis.set_label_coords(-0.17, 0.5)

    if showlabels:
        ax = fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.ylabel(f"Detections/{catalogue.freq}", fontsize=fontsize)
        ax.yaxis.set_label_coords(-0.2, 0.3)
        plt.subplots_adjust(left=0.2, right=0.95)

    if show:
        plt.show()
    else:
        plt.close()
    return fig


def view_specgram(X, insp_idx, n, o, fname_dataset, sample_index, figtitle,
                  nrows=2, ncols=2, figsize=(12,9), show=True):
    '''Plots selected spectrograms from input data.'''
    if not len(insp_idx) == nrows * ncols:
        raise ValueError('Subplot/sample number mismatch: check dimensions.')
    metadata = utils.get_metadata(insp_idx, sample_index, fname_dataset)
    fig = plt.figure(figsize=figsize, dpi=150)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    counter = 0
    for i in range(len(insp_idx)):
        ax = fig.add_subplot(gs[i])
        plt.imshow(torch.reshape(X[insp_idx[i],:,:,:], (n,o)), aspect='auto', interpolation="none")
        plt.gca().invert_yaxis()
        plt.xlabel('Time Bin')
        plt.ylabel('Frequency Bin')
        station = metadata[counter]['station']
        time_on = metadata[counter]['spec_start']
        plt.title(f'Station {station}\nTrigger: {time_on}\n'
                  f'Index: {sample_index[insp_idx[i]]}')
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
        x, x_r, z,
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
        'mathtext.default': 'regular',
        'text.latex.preamble': r'\usepackage{bm}',
    }
    plt.rcParams.update(rc_fonts)
    sample_idx = np.arange(0, len(disp_idx))
    metadata = utils.get_metadata(sample_idx, disp_idx, fname_dataset)
    X = x.detach().cpu().numpy()
    X_r = x_r.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    n, o = list(X.shape[2:])
    fig = plt.figure(figsize=(figsize[0],len(disp_idx)*figsize[1]), dpi=150)
    cmap = 'cmo.ice_r'
    heights = [4 for i in range(x.size()[0])]
    widths = [3, 0.5, 3]
    extent = [min(tvec), max(tvec), min(fvec), max(fvec)]
    arrow_yloc = 1.12
    gs = gridspec.GridSpec(nrows=x.size()[0], ncols=3, height_ratios=heights, width_ratios=widths, wspace=0.2, hspace=0.6)
    counter = 0
    for i in range(x.size()[0]):
        station = metadata[i]['station']
        time_on = metadata[counter]['spec_start']

        ax1 = fig.add_subplot(gs[counter,0])
        plt.imshow(np.reshape(X[i,:,:,:], (n,o)), cmap=cmap, extent=extent, aspect='auto', origin='lower', interpolation="none")
        plt.colorbar(orientation='vertical', pad=0)
        plt.clim(0,1)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Input\n' + r'$\bm{x}$', x=0.55)

        xy1 = (0.67, arrow_yloc)

        ax2 = fig.add_subplot(gs[counter,1])
        plt.imshow(np.expand_dims(z[i], 1), cmap=cmo.deep_r, aspect='auto', interpolation="none")
        # for j in range(z.shape[1]):
        #     plt.text(
        #         0,
        #         j,
        #         f"{z[i,j]:.1f}",
        #         backgroundcolor='w',
        #         ha='center',
        #         va='center',
        #         bbox=dict(boxstyle='square,pad=0', facecolor='w', edgecolor='w')
        #     )
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
        plt.imshow(np.reshape(X_r[i,:,:,:], (n,o)), cmap=cmap, extent=extent, aspect='auto', origin='lower', interpolation="none")
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
    df = pd.DataFrame(data=data, columns=["Class", "x", "y"])
    df["Class"] = df["Class"].astype('int').astype('category')

    fig = plt.figure(figsize=(6,8))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0])
    sns.scatterplot(data=df, x="x", y="y", hue="Class", palette=colors, alpha=0.2)
    plt.axis('off')
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.75), ncol=1)
    plt.title(title)

    ax2 = fig.add_subplot(gs[1])
    arr = plt.hist(labels+1, bins=np.arange(1, max(labels)+3, 1), histtype='bar', align='left', rwidth=0.8, color='k')
    plt.grid(axis='y', linestyle='--')
    plt.xticks(label_list+1, label_list+1)
    plt.ylim([0, 1.25 * max(counts)])
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Detections')
    plt.title(f'Class Assignments, N = {len(labels)}')

    N = counts.sum()
    def CtP(x):
        return 100 * x / N

    def PtC(x):
        return x * N / 100

    ax3 = ax2.secondary_yaxis('right', functions=(CtP, PtC))
    ax3.set_ylabel('\% of N')
    for i in range(len(np.unique(labels))):
        plt.text(arr[1][i], 1.05 * arr[0][i], str(int(arr[0][i])), ha='center')

    if show:
        plt.show()
    else:
        plt.close()
    return fig
