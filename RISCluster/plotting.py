import configparser
import csv
from datetime import datetime
import io
import os
import sys
sys.path.insert(0, '../RISCluster/')

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from processing import get_metadata
import utils
from networks import AEC, DCM

def cluster_gallery(model, labels, z_array, fname_dataset, device, centroids=None, p=1, show=False):
    text_trap = io.StringIO()
    # sys.stdout = text_trap
    model.eval()
    label_list, counts = np.unique(labels, return_counts=True)
    if centroids is not None:
        print(torch.from_numpy(centroids).dtype)
        X_c = model.decoder(torch.from_numpy(centroids).float().to(device))
    else:
        centroids = model.clustering.weights
        X_c = model.decoder(centroids)
        centroids = centroids.detach().cpu().numpy()

    fig = plt.figure(figsize=(len(label_list),12), dpi=100)
    gs_sup = gridspec.GridSpec(nrows=9, ncols=len(label_list), hspace=0.05, wspace=0.05)
    heights = [1, 4, 0.2]
    font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 6,
        }
    for l, label in enumerate(label_list):
        query = np.where(labels == label_list[l])[0]
        N = 8
        distance = utils.fractional_distance(centroids[l,:], z_array[query,:], p)
        # distance = np.linalg.norm(centroids[l,:] - z_array[query,:], ord=p, axis=1)
        image_index = np.argsort(distance)[0:N]

        transform = 'sample_norm_cent'
        dataset = utils.load_dataset(fname_dataset, image_index, send_message=False, transform=transform, **{"notqdm": True})
        dataloader = DataLoader(dataset, batch_size=N)
        X = []
        for batch in dataloader:
            X = batch.to(device)

        Z = model.encoder(X)

        with h5py.File(fname_dataset, 'r') as f:
            M = len(image_index)
            DataSpec = '/4s/Trace'
            dset = f[DataSpec]
            k = 351

            tr = np.empty([M, k])
            dset_arr = np.empty([k,])

            for i in range(M):
                dset_arr = dset[image_index[i], 25:-25]
                tr[i,:] = dset_arr

        gs_sub = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_sup[0,l], hspace=0, wspace=0, height_ratios=heights)
        ax = fig.add_subplot(gs_sub[0])
        plt.axis('off')
        ax = fig.add_subplot(gs_sub[1])
        plt.imshow(torch.squeeze(X_c[l]).detach().cpu().numpy(), aspect='auto', origin='lower')
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(f"Cluster {label}", size=5)
        if l == 0:
            plt.ylabel("Centroids", size=5)

        ax = fig.add_subplot(gs_sub[2])
        plt.imshow(np.expand_dims(centroids[l], 0), cmap='viridis', aspect='auto')
        plt.xticks([])
        plt.yticks([])

        for i in range(N):
            gs_sub = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_sup[i+1,l], hspace=0, wspace=0, height_ratios=heights)
            ax = fig.add_subplot(gs_sub[0])
            plt.plot(tr[i])
            plt.xticks([])
            plt.yticks([])
            ax = fig.add_subplot(gs_sub[1])
            plt.imshow(np.squeeze(X[i,:,:].detach().cpu().numpy()), aspect='auto', origin='lower')
            plt.text(0, 60, f"{image_index[i]}", fontdict=font)
            plt.xticks([])
            plt.yticks([])
            ax = fig.add_subplot(gs_sub[2])
            plt.imshow(np.expand_dims(Z[i].detach().cpu().numpy(), 0), cmap='viridis', aspect='auto')
            plt.xticks([])
            plt.yticks([])

    if show:
        plt.show()
    else:
        plt.close()
    # sys.stdout = sys.__stdout__
    return fig

def compare_images(
        model,
        disp,
        epoch,
        savepath,
        show=True
    ):
    model.eval()
    x_r, z = model(disp)
    figtitle = f'DCM Pre-training: Epoch {epoch}'
    n, o = list(disp.size()[2:])
    fig = view_specgram_training(
        disp,
        x_r, z,
        n, o,
        figtitle,
        figsize=(12,9),
        show=show
    )
    savepath_snap = savepath + '/snapshots/'
    if not os.path.exists(savepath_snap):
        os.makedirs(savepath_snap)
    figname = savepath_snap + f'AEC_Training_Epoch_{epoch:03d}.png'
    fig.savefig(figname)
    return fig

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

    fig = plt.figure(figsize=(3,2*len(centroids)), dpi=100)
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

def view_clusters(pca2d, labels):
    fig = plt.figure(figsize=(6,6), dpi=300)
    sns.scatterplot(pca2d[:,0], pca2d[:,1], hue=labels, palette='Set1', alpha=0.2)
    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    fig.tight_layout()
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

        fig = plt.figure(figsize=(12,9), dpi=300)
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
            fig.savefig(f'{savepath}/Label{label_list[l]:02d}_Examples.png')

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

def view_specgram_training(x, x_r, z, n, o, figtitle,
                           figsize=(12,9), show=True):
    X = x.detach().cpu().numpy()
    X_r = x_r.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(nrows=3, ncols=4)
    counter = 0
    for i in range(x.size()[0]):
        ax = fig.add_subplot(gs[0,counter])
        plt.imshow(np.reshape(X[i,:,:,:], (n,o)), aspect='auto')
        plt.gca().invert_yaxis()
        plt.xlabel('Time Bin')
        plt.ylabel('Frequency Bin')
        if counter == 0:
            plt.figtext(-0.01, 0.62, 'Original Spectrograms', rotation='vertical',
                        fontweight='bold')

        ax = fig.add_subplot(gs[1,counter])
        plt.imshow(np.expand_dims(z[i], 1), aspect='auto')

        ax = fig.add_subplot(gs[2,counter])
        plt.imshow(np.reshape(X_r[i,:,:,:], (n,o)), aspect='auto')
        plt.gca().invert_yaxis()
        plt.xlabel('Time Bin')
        plt.ylabel('Frequency Bin')
        if counter == 0:
            plt.figtext(-0.01, 0.15, 'Reconstructed Spectrograms',
                        rotation='vertical', fontweight='bold')
        counter += 1

    fig.suptitle(figtitle, size=18, weight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
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

def view_TSNE(results, labels, title, show=False):
    fig = plt.figure(figsize=(6,9))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[4, 3])

    ax = fig.add_subplot(gs[0])
    sns.scatterplot(results[:, 0], results[:, 1], hue=labels, palette='Set1', alpha=0.2)
    plt.axis('off')
    plt.title('T-SNE Results for GMM Centroids')

    ax = fig.add_subplot(gs[1])
    arr = plt.hist(labels, bins=np.arange(0, max(labels)+2, 1), histtype='bar', align='left', rwidth=0.8)
    plt.grid(axis='y', linestyle='--')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Detections')
    plt.title('Histogram of Cluster Assignments')

    _, counts = np.unique(labels, return_counts=True)
    N = counts.sum()

    def CtP(x):
        return 100 * x / N

    def PtC(x):
        return x * N / 100

    ax2 = ax.secondary_yaxis('right', functions=(CtP, PtC))
    ax2.set_ylabel('Percentage of Detections')
    for i in range(len(np.unique(labels))):
        plt.text(arr[1][i], arr[0][i]+10, str(int(arr[0][i])), ha='center')

    if show:
        plt.show()
    else:
        plt.close()
    return fig
