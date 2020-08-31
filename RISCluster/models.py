from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime
import os
import random
import shutil
import sys
sys.path.insert(0, '../RISCluster/')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import torch
# import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

import importlib as imp
import plotting
imp.reload(plotting)
import utils
imp.reload(utils)

def pretrain_DCM(
        model,
        dataloaders,
        criteria,
        optimizer,
        batch_size,
        lr,
        parameters
    ):
    '''
    Function facilitates pre-training (i.e., training of AEC) of the DCM model.
    # Arguments:
        model: PyTorch model instance
        dataloader: PyTorch dataloader instance
        criteria: PyTorch loss function instances
        optimizer: PyTorch optimizer instance
        batch_size: Batch size used in calculations
        lr: Learning rate
        parameters: Additional variables
    # Returns:
        model: Trained model weights are saved to disk.
        tb: Tensorboard file recording various parameters saved to disk.
        disp: Input/output spectrograms are saved to disk.
    '''
    tic = datetime.now()
    print('Commencing pre-training...')

    n_epochs = parameters['n_epochs']
    show = parameters['show']
    device = parameters['device']
    mode = parameters['mode']
    savepath_exp = parameters['savepath']
    show = parameters['show']
    early_stopping = parameters['early_stopping']
    patience = parameters['patience']

    savepath_run, serial_run = utils.init_output_env(
        savepath_exp,
        mode,
        **{
        'batch_size': batch_size,
        'lr': lr
        }
    )

    criterion_mse = criteria[0]
    criterion_mae = criteria[1]

    tra_loader = dataloaders[0]
    val_loader = dataloaders[1]
    M_tra = len(tra_loader.dataset)
    M_val = len(val_loader.dataset)

    images = next(iter(tra_loader))

    disp_idx = sorted(np.random.randint(0, images.size(0), 4))
    disp = images[disp_idx]

    tb = SummaryWriter(log_dir=savepath_run)
    fig = plotting.compare_images(
        model,
        disp.to(device),
        0,
        savepath_run,
        show
    )
    tb.add_figure('TrainingProgress', fig, global_step=0, close=True)

    if early_stopping:
        savepath_chkpnt = f'{savepath_run}/tmp/'
        if not os.path.exists(savepath_chkpnt):
            os.makedirs(savepath_chkpnt)
        best_val_loss = 10000

    finished = False
    for epoch in range(n_epochs):
        print('-' * 100)
        print(
            f'Epoch [{epoch+1}/{n_epochs}] | '
            f'Batch Size = {batch_size} | LR = {lr}'
        )
        # ==== Training Loop: =================================================
        model.train(True)

        running_tra_mse = 0.0
        running_tra_mae = 0.0
        running_size = 0

        pbar_tra = tqdm(
            tra_loader,
            leave=True,
            desc="  Training",
            unit="batch",
            postfix={
                "MAE": "%.6f" % 0.0,
                "MSE": "%.6f" % 0.0
            },
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for batch in pbar_tra:
            x = batch.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                x_rec, _ = model(x)
                loss_mse = criterion_mse(x_rec, x)
                loss_mae = criterion_mae(x_rec, x)
                loss_mse.backward()
                optimizer.step()

            running_tra_mse += loss_mse.cpu().detach().numpy() * x.size(0)
            running_tra_mae += loss_mae.cpu().detach().numpy() * x.size(0)
            running_size += x.size(0)

            pbar_tra.set_postfix(
                MAE = f"{(running_tra_mae / running_size):.4e}",
                MSE = f"{(running_tra_mse / running_size):.4e}"
            )

        epoch_tra_mse = running_tra_mse / M_tra
        epoch_tra_mae = running_tra_mae / M_tra
        tb.add_scalar('Training MSE', epoch_tra_mse, epoch)
        tb.add_scalar('Training MAE', epoch_tra_mae, epoch)

        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        if (epoch % 5) == 0 and not (epoch == 0):
            fig = plotting.compare_images(
                model,
                disp.to(device),
                epoch,
                savepath_run,
                show
            )
            tb.add_figure('TrainingProgress', fig, global_step=epoch, close=True)
        # ==== Validation Loop: ===============================================
        model.train(False)

        running_val_mse = 0.0
        running_val_mae = 0.0
        running_size = 0

        pbar_val = tqdm(
            val_loader,
            leave=True,
            desc="Validation",
            unit="batch",
            postfix={
                "MSE": "%.6f" % 0.0,
                "MAE": "%.6f" % 0.0
            },
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for batch in pbar_val:
            model.eval()
            with torch.no_grad():
                x = batch.to(device)
                x_rec, _ = model(x)
                loss_mse = criterion_mse(x_rec, x)
                loss_mae = criterion_mae(x_rec, x)

            running_val_mse += loss_mse.cpu().detach().numpy() * x.size(0)
            running_val_mae += loss_mae.cpu().detach().numpy() * x.size(0)
            running_size += x.size(0)

            pbar_val.set_postfix(
                MSE = f"{(running_val_mse / running_size):.4e}",
                MAE = f"{(running_val_mae / running_size):.4e}"
            )

        epoch_val_mse = running_val_mse / M_val
        epoch_val_mae = running_val_mae / M_val
        tb.add_scalar('Validation MSE', epoch_val_mse, epoch)
        tb.add_scalar('Validation MAE', epoch_val_mae, epoch)

        if early_stopping:

            if epoch_val_mse < best_val_loss:
                strikes = 0
                best_val_loss = epoch_val_mse
                fname = f'{savepath_chkpnt}AEC_Best_Weights.pt'
                torch.save(model.state_dict(), fname)
            else:
                strikes += 1

            if epoch > patience and strikes > patience:
                print('Stopping Early.')
                finished = True
                break

    tb.add_hparams(
        {'Batch Size': batch_size, 'LR': lr},
        {
            'hp/Training MSE': epoch_tra_mse,
            'hp/Validation MSE': epoch_val_mse
        }
    )
    tb.close()
    if early_stopping and (finished == True or epoch == n_epochs-1):
        src_file = f'{savepath_chkpnt}AEC_Best_Weights.pt'
        dst_file = f'{savepath_run}/AEC_Params_{serial_run}.pt'
        shutil.move(src_file, dst_file)
    else:
        fname = f'{savepath_run}/AEC_Params_ {serial_run}.pt'
        torch.save(model.state_dict(), fname)
    print('AEC parameters saved.')

    toc = datetime.now()
    print(f'Pre-training complete at {toc}; time elapsed = {toc-tic}.')
    return model, tb

def train_DCM(
        model,
        dataloader,
        criteria,
        optimizer,
        n_clusters,
        batch_size,
        lr,
        gamma,
        tol,
        parameters
        ):
    '''
    Function facilitates training (i.e., clustering and fine-tuning of AEC) and
    of the DCM model.
    # Arguments:
        model: PyTorch model instance
        dataloader: PyTorch dataloader instance
        criteria: PyTorch loss function instances
        optimizer: PyTorch optimizer instance
        n_epochs: Number of epochs over which to fine-tune and cluster
        params: Administrative variables
    # Returns:
        model: Trained PyTorch model instance
    '''
    tic = datetime.now()
    print('Commencing training...')

    # Unpack parameters:
    device = parameters['device']
    n_epochs = parameters['n_epochs']
    update_interval = parameters['update_interval']
    savepath_exp = parameters['savepath']
    show = parameters['show']
    mode = parameters['mode']
    loadpath = parameters['saved_weights']
    savepath_run, serial_run = utils.init_output_env(
        savepath_exp,
        mode,
        **{
        'n_clusters': n_clusters,
        'batch_size': batch_size,
        'lr': lr,
        'gamma': gamma,
        'tol': tol
        }
    )

    model.load_state_dict(
        torch.load(loadpath, map_location=device), strict=False
    )
    model.eval()

    criterion_mse = criteria[0]
    criterion_kld = criteria[1]

    M = len(dataloader.dataset)

    tb = SummaryWriter(log_dir = savepath_run)

    # Initialize Clusters:
    print('Initiating clusters with k-means...')
    labels, centroids = kmeans(model, copy.deepcopy(dataloader), device)
    cluster_centers = torch.from_numpy(centroids).to(device)
    with torch.no_grad():
        model.state_dict()["clustering.weights"].copy_(cluster_centers)
    print('Clusters initiated.')
    # Initialize Target Distribution:
    q, labels_prev = predict_labels(model, dataloader, device)
    p = target_distribution(q)

    fig1, fig2 = analyze_clustering(model, dataloader, labels_prev, device, 0)
    tb.add_figure('Centroids',fig1, global_step=0, close=True)
    tb.add_figure('TSNE', fig2, global_step=0, close=True)

    n_iter = 1
    finished = False
    for epoch in range(n_epochs):
        print('-' * 110)
        print(
            f'Epoch [{epoch+1}/{n_epochs}] | '
            f'# Clusters = {n_clusters} | '
            f'Batch Size = {batch_size} | '
            f'LR = {lr} | '
            f'gamma = {gamma} | '
            f'tol = {tol}'
        )

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0
        running_size = 0
        # batch_num = 0

        pbar = tqdm(
            dataloader,
            leave=True,
            unit="batch",
            postfix={
                "MSE": "%.6f" % 0.0,
                "KLD": "%.6f" % 0.0,
                "Loss": "%.6f" % 0.0
            },
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        # Iterate over data:
        for batch_num, batch in enumerate(pbar):
            x = batch.to(device)
            # Uptade target distribution, check performance
            if (batch_num % update_interval == 0) and not \
                (batch_num == 0 and epoch == 0):
                q, labels = predict_labels(model, dataloader, device)
                p = target_distribution(q)
                # check stop criterion
                delta_label = np.sum(labels != labels_prev).astype(np.float32) \
                              / labels.shape[0]
                tb.add_scalar('delta', delta_label, n_iter)
                labels_prev = np.copy(labels)
                if delta_label < tol:
                    print('Stop criterion met, training complete.')
                    finished = True
                    break

            tar_dist = p[running_size:(running_size + x.size(0)), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)

            # zero the parameter gradients
            model.train()
            optimizer.zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                q, x_rec, _ = model(x)
                loss_rec = criterion_mse(x_rec, x)
                loss_clust = gamma * criterion_kld(torch.log(q), tar_dist) / x.size(0)
                loss = loss_rec + loss_clust
                loss.backward()
                optimizer.step()

            running_size += x.size(0)
            running_loss += loss.cpu().detach().numpy() * x.size(0)
            running_loss_rec += loss_rec.cpu().detach().numpy() * x.size(0)
            running_loss_clust += loss_clust.cpu().detach().numpy() * x.size(0)

            accum_loss = running_loss / running_size
            accum_loss_rec = running_loss_rec / running_size
            accum_loss_clust = running_loss_clust / running_size

            pbar.set_postfix(
                MSE = f"{accum_loss_rec:.4e}",
                KLD = f"{accum_loss_clust:.4e}",
                Loss = f"{accum_loss:.4e}"
            )

            tb.add_scalars(
                'Losses',
                {
                    'Loss': accum_loss,
                    'MSE': accum_loss_rec,
                    'KLD': accum_loss_clust
                },
                n_iter
            )
            tb.add_scalar('Loss', accum_loss, n_iter)
            tb.add_scalar('MSE', accum_loss_rec, n_iter)
            tb.add_scalar('KLD', accum_loss_clust, n_iter)
            for name, weight in model.named_parameters():
                tb.add_histogram(name, weight, n_iter)
                tb.add_histogram(f'{name}.grad', weight.grad, n_iter)

            if (batch_num % update_interval == 0) and not \
                (batch_num == 0 and epoch == 0):
                fig1, fig2 = analyze_clustering(model, dataloader, labels, device, epoch)
                tb.add_figure('Centroids',fig1, global_step=n_iter, close=True)
                tb.add_figure('TSNE', fig2, global_step=n_iter, close=True)

            n_iter += 1

        if finished:
            break

    tb.add_hparams(
        {'Clusters': n_clusters, 'Batch Size': batch_size, 'LR': lr, 'gamma': gamma, 'tol': tol},
        {
            'hp/MSE': accum_loss_rec,
            'hp/KLD': accum_loss_clust,
            'hp/Loss': accum_loss
        }
    )
    tb.close()
    fname = f'{savepath_run}/DCM_Params_{serial_run}.pt'
    torch.save(model.state_dict(), fname)
    print('DCM parameters saved.')
    toc = datetime.now()
    print(f'Pre-training complete at {toc}; time elapsed = {toc-tic}.')
    return model

def predict_DCM(model, dataloader, idx_smpl, parameters):
    device = parameters['device']
    savepath_exp = parameters['savepath']
    serial_exp = parameters['serial']
    mode = parameters['mode']
    loadpath = parameters['saved_weights']
    n_clusters = parameters['n_clusters']
    max_workers = parameters['max_workers']

    savepath_run, serial_run = utils.init_output_env(
        savepath_exp,
        mode,
        **{'n_clusters': n_clusters}
    )

    model.load_state_dict(torch.load(loadpath, map_location=device))
    model.eval()

    pbar = tqdm(
        dataloader,
        leave=True,
        desc="Saving cluster labels",
        unit="batch",
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    label_list = []

    running_size = 0
    # counter = 0
    for batch_num, batch in enumerate(pbar):
        x = batch.to(device)
        # q, x_rec, z = model(x)
        q, _, _ = model(x)
        label = torch.argmax(q, dim=1)
        A = [
                {
                    # 'x': x[i].cpu().detach().numpy(),
                    'label': label[i].cpu().detach().numpy(),
                    # 'x_rec': x_rec[i].cpu().detach().numpy(),
                    # 'z': z[i].cpu().detach().numpy(),
                    'idx': idx_smpl[running_size:(running_size + x.size(0))][i],
                    # 'savepath': savepath_run[int(label[i])]
                } for i in range(x.size(0))]
        # print('--------------------------------------------------------------')
        # print(f'Saving outputs for Batch {batch_num}:')
        utils.save_labels(
            [{k: v for k, v in d.items() if \
                (k == 'idx' or k == 'label')} for d in A],
            savepath_exp,
            serial_exp
        )
        # print('Saving spectrograms to file...')
        # # Parallel Implementation
        # with ProcessPoolExecutor(max_workers=max_workers) as exec:
        #     futures = [exec.submit(plotting.save_DCM_output, **a) for a in A]
        #     kwargs = {
        #         'total': len(futures),
        #         'unit': 'it',
        #         'unit_scale': True,
        #         'leave': True,
        #         'bar_format': '{l_bar}{bar:20}{r_bar}{bar:-20b}'
        #     }
        #     for future in tqdm(as_completed(futures), **kwargs):
        #         future.result()

        # Serial Implementation:
        # for i in tqdm(range(x.size(0))):
        #     plotting.save_DCM_output(
        #         A[i]['x'],
        #         A[i]['label'],
        #         A[i]['x_rec'],
        #         A[i]['z'],
        #         A[i]['idx'],
        #         A[i]['savepath'],
        #     )
        # - Save stats for histograms
        running_size += x.size(0)

# K-means clusters initialisation
def kmeans(model, dataloader, device):
    '''
    Initiate clusters using K-means algorithm.
    # Arguments:
        model: PyTorch model instance
        dataloader: PyTorch dataloader instance
        device: PyTorch device object ('cpu' or 'gpu')
    # Inputs:
        n_clusters: Number of clusters, set during model construction.
        n_init: Number of iterations for initialization.
    # Returns:
        labels: Sample-wise cluster assignment
        centroids: Sample-wise cluster centroids
    '''
    km = KMeans(n_clusters=model.n_clusters, n_init=100)
    z_array = None
    model.eval()
    for batch in dataloader:
        x = batch.to(device)
        _, _, z = model(x)
        if z_array is not None:
            z_array = np.concatenate((z_array, z.cpu().detach().numpy()), 0)
        else:
            z_array = z.cpu().detach().numpy()

    km.fit_predict(z_array)

    labels = km.labels_
    centroids = km.cluster_centers_
    return labels, centroids

def gmm(model, dataloader, device):
    '''
    Initiate clusters using GMM algorithm.
    # Arguments:
        model: PyTorch model instance
        dataloader: PyTorch dataloader instance
        device: PyTorch device object ('cpu' or 'gpu')
    # Inputs:
        n_clusters: Number of clusters, set during model construction.
    # Returns:
        labels: Sample-wise cluster assignment
        centroids: Sample-wise cluster centroids
    '''
    gmm = GaussianMixture(n_components=model.n_clusters)
    z_array = None
    model.eval()
    for batch in dataloader:
        x = batch.to(device)
        _, _, z = model(x)
        if z_array is not None:
            z_array = np.concatenate((z_array, z.cpu().detach().numpy()), 0)
        else:
            z_array = z.cpu().detach().numpy()

    labels = gmm.fit_predict(z_array)
    centroids = gmm.means_
    return labels, centroids

# def pca(labels, model, dataloader, device, tb, counter):
#     z_array = None
#     model.eval()
#     for batch in dataloader:
#         x = batch.to(device)
#         _, _, z = model(x)
#         if z_array is not None:
#             z_array = np.concatenate((z_array, z.cpu().detach().numpy()), 0)
#         else:
#             z_array = z.cpu().detach().numpy()
#
#     row_max = z_array.max(axis=1)
#     z_array /= row_max[:, np.newaxis]
#
#     pca2 = PCA(n_components=model.n_clusters).fit(z_array)
#     pca2d = pca2.transform(z_array)
#     fig = plotting.view_clusters(pca2d, labels)
#     tb.add_figure('PCA_Z', fig, global_step=counter, close=True)


def predict_labels(model, dataloader, device):
    '''
    Function takes input data from dataloader, feeds through encoder layers,
    and returns predicted soft- and hard-assigned cluster labels.
    # Arguments:
        model: PyTorch model instance
        dataloader: PyTorch dataloader instance
        device: PyTorch device object ('cpu' or 'gpu')
    # Returns:
        q_array [n_samples, n_clusters]: Soft assigned label probabilities
        labels [n_samples,]: Hard assigned label based on max of q_array
    '''
    q_array = None
    model.eval()
    for batch in dataloader:
        x = batch.to(device)
        q, _, _ = model(x)
        if q_array is not None:
            q_array = np.concatenate((q_array, q.cpu().detach().numpy()), 0)
        else:
            q_array = q.cpu().detach().numpy()

    labels = np.argmax(q_array.data, axis=1)
    return np.round(q_array, 5), labels

def target_distribution(q):
    '''
    Compute the target distribution p, given soft assignements, q. The target
    distribtuion is generated by giving more weight to 'high confidence'
    samples - those with a higher probability of being a signed to a certain
    cluster.  This is used in the KL-divergence loss function.
    # Arguments
        q: Soft assignement probabilities - Probabilities of each sample being
           assigned to each cluster.
    # Input:
        2D array of shape [n_samples, n_features].
    # Output:
        2D array of shape [n_samples, n_features].
    '''
    p = q ** 2 / np.sum(q, axis=0)
    p = np.transpose(np.transpose(p) / np.sum(p, axis=1))
    return np.round(p, 5)

def analyze_clustering(model, dataloader, labels, device, epoch):
    # Step 1: Show Centroid outputs
    centroids = model.clustering.weights
    X_r = model.decoder(centroids)
    fig1 = plotting.view_centroid_output(centroids, X_r, f'Centroid Reconstructions - Epoch {epoch}')
    # Step 2: Show t-SNE & labels
    z_array = None
    model.eval()
    for batch in tqdm(dataloader):
        x = batch.to(device)
        _, z = model(x)
        if z_array is not None:
            z_array = np.concatenate((z_array, z.cpu().detach().numpy()), 0)
        else:
            z_array = z.cpu().detach().numpy
    data = z_array.astype('float64')

    results = TSNE(n_components=2, perplexity=50, learning_rate=200, n_jobs=16, verbose=0).fit_transform(data)
    fig2 = plt.figure()
    sns.scatterplot(results[:, 0], results[:, 1], hue=labels, palette='Set1', alpha=0.2)
    plt.title(f'T-SNE Results - Epoch {epoch}')
    return fig1, fig2
