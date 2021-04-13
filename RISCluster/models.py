#!/usr/bin/env python3

'''Contains necessary functions, routines, plotting wrappers, and data
recording for DEC model initialization, training, validation, and inference.

William Jenkins, wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography, UC San Diego
January 2021
'''

from datetime import datetime
import threading
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
if sys.platform == 'darwin':
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
elif sys.platform == 'linux':
    from cuml import KMeans, TSNE
    import cupy
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from RISCluster import plotting, utils


def pretrain(
        model,
        dataloaders,
        criteria,
        optimizer,
        batch_size,
        lr,
        parameters
    ):
    '''Pre-trains DEC model (i.e., trains AEC).

    Parameters
    ----------
    model : PyTorch model instance
        Model with untrained parameters

    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    criteria : PyTorch loss function instances
        Error metrics

    optimizer : PyTorch optimizer instance

    batch_size : int
        Batch size used in calculations

    lr : float
        Controls initial learning rate for gradient descent.

    parameters : dict
        Additional experiment parameters/hyperparameters

    Returns
    -------
    model : PyTorch model instance
        Model with trained parameters

    Outputs to Disk
    ---------------
    Tensorboard Summary Writer : Records training and validation

    Matplotlib Figures : Prints spectrogram reconstructions to disk.
    '''
    tic = datetime.now()
    print('Commencing pre-training...')

    n_epochs = parameters['n_epochs']
    show = parameters['show']
    device = parameters['device']
    mode = parameters['mode']
    savepath_exp = parameters['savepath']
    fname_dataset = parameters['fname_dataset']
    show = parameters['show']
    early_stopping = parameters['early_stopping']
    patience = parameters['patience']
    km_metrics = parameters['km_metrics']
    disp_index = parameters['img_index']
    disp_index = [int(i) for i in disp_index.split(',')]
    tbpid = parameters['tbpid']

    savepath_run, serial_run, savepath_chkpnt = utils.init_output_env(
        savepath_exp,
        mode,
        **{
        'batch_size': batch_size,
        'lr': lr
        }
    )

    criterion_mse = criteria[0]
    tra_loader = dataloaders[0]
    val_loader = dataloaders[1]

    tb = SummaryWriter(log_dir=savepath_run)
    if tbpid is not None:
        tb.add_text(
            "Tensorboard PID",
            f"To terminate this TB instance, kill PID: {tbpid}",
            global_step=None
        )
    tb.add_text("Path to Saved Outputs", savepath_run, global_step=None)
    fig = plotting.compare_images(
        model,
        0,
        disp_index,
        fname_dataset,
        device,
        savepath=savepath_run,
        show=show,
    )
    tb.add_figure(
        'TrainingProgress',
        fig,
        global_step=0,
        close=True
    )

    if early_stopping:
        best_val_loss = 10000

    epochs = list()
    tra_losses = list()
    val_losses = list()
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
        running_size = 0

        pbar_tra = tqdm(
            tra_loader,
            leave=True,
            desc="  Training",
            unit="batch",
            postfix={"MSE": "%.6f" % 0.0},
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for batch in pbar_tra:
            _, batch = batch
            x = batch.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                x_rec, _ = model(x)
                loss_mse = criterion_mse(x_rec, x)
                loss_mse.backward()
                optimizer.step()

            running_tra_mse += loss_mse.cpu().detach().numpy() * x.size(0)
            running_size += x.size(0)

            pbar_tra.set_postfix(
                MSE = f"{(running_tra_mse / running_size):.4e}"
            )

        epoch_tra_mse = running_tra_mse / len(tra_loader.dataset)
        tb.add_scalar('Training MSE', epoch_tra_mse, epoch)

        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        if (epoch % 5) == 0 and not (epoch == 0):
            fig = plotting.compare_images(
                model,
                epoch,
                disp_index,
                fname_dataset,
                device,
                savepath=savepath_run,
                show=show,
            )
            tb.add_figure(
                'TrainingProgress',
                fig,
                global_step=epoch,
                close=True
            )
        # ==== Validation Loop: ===============================================
        model.train(False)

        running_val_mse = 0.0
        running_size = 0

        pbar_val = tqdm(
            val_loader,
            leave=True,
            desc="Validation",
            unit="batch",
            postfix={"MSE": "%.6f" % 0.0},
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for batch in pbar_val:
            _, batch = batch
            x = batch.to(device)
            model.eval()
            with torch.no_grad():
                # x = batch.to(device)
                x_rec, _ = model(x)
                loss_mse = criterion_mse(x_rec, x)

            running_val_mse += loss_mse.cpu().detach().numpy() * x.size(0)
            running_size += x.size(0)

            pbar_val.set_postfix(
                MSE = f"{(running_val_mse / running_size):.4e}"
            )

        epoch_val_mse = running_val_mse / len(val_loader.dataset)
        tb.add_scalar('Validation MSE', epoch_val_mse, epoch)

        epochs, tra_losses, val_losses = utils.add_to_history(
            [epochs, tra_losses, val_losses],
            [epoch, epoch_tra_mse, epoch_val_mse]
        )

        if early_stopping:
            if epoch_val_mse < best_val_loss:
                strikes = 0
                best_val_loss = epoch_val_mse
                fname = f'{savepath_chkpnt}/AEC_Best_Weights.pt'
                torch.save(model.state_dict(), fname)
            else:
                if epoch == 0:
                    strikes = 1
                else:
                    strikes += 1

            if epoch > patience and strikes > patience:
                print('Stopping Early.')
                finished = True
                break
        else:
            fname = f'{savepath_chkpnt}/AEC_Params_{epoch:03d}.pt'
            torch.save(model.state_dict(), fname)

    _ = utils.save_history(
        {
            'Epoch': epochs,
            'Training Loss': tra_losses,
            'Validation Loss': val_losses
        },
        f"{savepath_run}/AEC_history.csv"
    )
    fig2 = plotting.view_history_AEC(f"{savepath_run}/AEC_history.csv")
    fig2.savefig(f"{savepath_run}/AEC_history.png", dpi=300, facecolor='w')
    tb.add_hparams(
        {'Batch Size': batch_size, 'LR': lr},
        {
            'hp/Training MSE': epoch_tra_mse,
            'hp/Validation MSE': epoch_val_mse
        }
    )
    fig = plotting.compare_images(
        model,
        epoch,
        disp_index,
        fname_dataset,
        device,
        savepath=savepath_run,
        show=show
    )
    tb.add_figure(
        'TrainingProgress',
        fig,
        global_step=epoch,
        close=True
    )
    fname = f'{savepath_run}/AEC_Params_Final.pt'
    if early_stopping and (finished == True or epoch == n_epochs-1):
        src_file = f'{savepath_chkpnt}/AEC_Best_Weights.pt'
        shutil.move(src_file, fname)
    else:
        torch.save(model.state_dict(), fname)
    tb.add_text("Path to Saved Weights", fname, global_step=None)
    print('AEC parameters saved.')

    toc = datetime.now()
    print(f'Pre-training complete at {toc}; time elapsed = {toc-tic}.')

    if km_metrics:
        klist = parameters['klist']
        klist = np.arange(int(klist.split(',')[0]), int(klist.split(',')[1])+1)
        print('-' * 62)
        print("Calculating optimal cluster size...")
        inertia, silh, gap_g, gap_u = kmeans_metrics(
            tra_loader,
            model,
            device,
            klist
        )
        fig = plotting.view_cluster_stats(klist, inertia, silh, gap_g, gap_u)
        plt.savefig(f'{savepath_run}/KMeans_Metrics.png', dpi=300)
        print("K-means statistics complete; figure saved.")
        tb.add_figure('K-Means Metrics', fig, global_step=None, close=True)

    tb.close()
    return model


def train(
        model,
        dataloader,
        criteria,
        optimizer,
        n_clusters,
        batch_size,
        lr,
        gamma,
        tol,
        index_tra,
        parameters
    ):
    '''Trains DEC model & performs clustering.

    Parameters
    ----------
    model : PyTorch model instance
        Model with untrained parameters

    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    criteria : PyTorch loss function instances
        Error metrics

    optimizer : PyTorch optimizer instance

    n_clusters : int
        Number of clusters

    batch_size : int
        Batch size used in calculations

    lr : float
        Controls initial learning rate for gradient descent.

    gamma : float
        Hyperparameter that controls contribution of clustering loss to total
        loss.

    tol : float
        Threshold at which DEC stops.

    index_tra: array
        Indeces of data samples to be used for DEC training.

    parameters : dict
        Additional experiment parameters/hyperparameters

    Returns
    -------
    model : PyTorch model instance
        Model with trained parameters

    Outputs to Disk
    ---------------
    Tensorboard Summary Writer : Records training and clustering

    Matplotlib Figures : Prints DEC figures to disk.
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
    fname_dataset = parameters['fname_dataset']
    tbpid = parameters['tbpid']
    init = parameters['init']
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
    fignames = [
        'T-SNE',
        'Gallery',
        'DistMatrix',
        'LatentSpace',
        'CDF',
        'PDF'
    ]
    figpaths = [utils.make_dir(fignames[i], savepath_run) for i in range(len(fignames))]

    model.load_state_dict(
        torch.load(loadpath, map_location=device), strict=False
    )
    model.eval()

    criterion_mse = criteria[0]
    criterion_kld = criteria[1]

    M = len(dataloader.dataset)
    if update_interval == -1:
        update_interval = int(np.ceil(M / (batch_size * 2)))

    tb = SummaryWriter(log_dir = savepath_run)
    if tbpid is not None:
        tb.add_text(
            "Tensorboard PID",
            f"To terminate this TB instance, kill PID: {tbpid}",
            global_step=None
        )
    tb.add_text("Path to Saved Outputs", savepath_run, global_step=None)
    # Initialize Clusters:
    if init == "kmeans": # K-Means Initialization:
        print('Initiating clusters with k-means...', end="", flush=True)
        labels_prev, centroids = kmeans(model, dataloader, device)
    elif init == "gmm": # GMM Initialization:
        print('Initiating clusters with GMM...', end="", flush=True)
        labels_prev, centroids = gmm(model, dataloader, device)
        # labels_prev =
        # centroids = np.random.randn()
    # elif init == "kmeds": # K-Medoids Initialization:
    #     print('Initiating clusters with k-medoids...', end="", flush=True)
    #     labels_prev, centroids = kmeds(model, dataloader, device)
    cluster_centers = torch.from_numpy(centroids).to(device)
    with torch.no_grad():
        model.state_dict()["clustering.weights"].copy_(cluster_centers)
    fname = f'{savepath_run}/DEC_Params_Initial.pt'
    torch.save(model.state_dict(), fname)
    print('complete.')
    # Initialize Target Distribution:
    q, _, z_array0 = infer(dataloader, model, device) # <-- The CUDA problem occurs in here
    p = target_distribution(q)
    epoch = 0
    tsne_results = tsne(z_array0)
    plotargs = (
            fignames,
            figpaths,
            tb,
            model,
            dataloader,
            device,
            fname_dataset,
            index_tra,
            z_array0,
            z_array0,
            labels_prev,
            labels_prev,
            centroids,
            centroids,
            tsne_results,
            epoch,
            show
    )
    plot_process = threading.Thread(target=plotting.plotter_mp, args=plotargs)
    plot_process.start()

    iters = list()
    rec_losses = list()
    clust_losses = list()
    total_losses = list()

    deltas_iter = list()
    deltas = list()

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
            _, batch = batch
            x = batch.to(device)
            # Update target distribution, check performance
            if (batch_num % update_interval == 0) and not \
                (batch_num == 0 and epoch == 0):
                q, labels, _ = infer(dataloader, model, device)
                p = target_distribution(q)
                # check stop criterion
                delta_label = np.sum(labels != labels_prev).astype(np.float32)\
                    / labels.shape[0]
                deltas_iter, deltas = utils.add_to_history(
                    [deltas_iter, deltas],
                    [n_iter, delta_label]
                )
                deltas.append(delta_label)
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
                loss_clust = gamma * criterion_kld(torch.log(q), tar_dist) \
                    / x.size(0)
                loss = loss_rec + loss_clust
                loss.backward()
                optimizer.step()

            running_size += x.size(0)
            running_loss += loss.detach().cpu().numpy() * x.size(0)
            running_loss_rec += loss_rec.detach().cpu().numpy() * x.size(0)
            running_loss_clust += loss_clust.detach().cpu().numpy() * x.size(0)

            accum_loss = running_loss / running_size
            accum_loss_rec = running_loss_rec / running_size
            accum_loss_clust = running_loss_clust / running_size

            pbar.set_postfix(
                MSE = f"{accum_loss_rec:.4e}",
                KLD = f"{accum_loss_clust:.4e}",
                Loss = f"{accum_loss:.4e}"
            )
            iters, rec_losses, clust_losses, total_losses = \
                utils.add_to_history(
                    [iters, rec_losses, clust_losses, total_losses],
                    [n_iter, accum_loss_rec, accum_loss_clust, accum_loss]
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

            n_iter += 1

        if ((epoch % 4 == 0) and not (epoch == 0)) or finished:
            _, _, z_array1 = infer(dataloader, model, device)
            tsne_results = tsne(z_array1)
            plotargs = (
                    fignames,
                    figpaths,
                    tb,
                    model,
                    dataloader,
                    device,
                    fname_dataset,
                    index_tra,
                    z_array0,
                    z_array1,
                    labels_prev,
                    labels,
                    centroids,
                    model.clustering.weights.detach().cpu().numpy(),
                    tsne_results,
                    epoch,
                    show
            )
            plot_process = threading.Thread(
                target=plotting.plotter_mp,
                args=plotargs
            )
            plot_process.start()

        if finished:
            break

    _ = utils.save_history(
        {
            'Iteration': iters,
            'Reconstruction Loss': rec_losses,
            'Clustering Loss': clust_losses,
            'Total Loss': total_losses
        },
        f"{savepath_run}/DEC_history.csv"
    )
    _ = utils.save_history(
        {
            'Iteration': deltas_iters,
            'Delta': deltas
        },
        f"{savepath_run}/Delta_history.csv"
    )
    tb.add_hparams(
        {
            'Clusters': n_clusters,
            'Batch Size': batch_size,
            'LR': lr,
            'gamma': gamma,
            'tol': tol},
        {
            'hp/MSE': accum_loss_rec,
            'hp/KLD': accum_loss_clust,
            'hp/Loss': accum_loss
        }
    )

    fname = f'{savepath_run}/DEC_Params_Final.pt'
    torch.save(model.state_dict(), fname)
    tb.add_text("Path to Saved Weights", fname, global_step=None)
    tb.close()
    print('DEC parameters saved.')
    toc = datetime.now()
    print(f'Pre-training complete at {toc}; time elapsed = {toc-tic}.')
    return model


def predict(model, dataloader, parameters):
    '''Run DEC model in inference mode.

    Parameters
    ----------
    model : PyTorch model instance
        Model with trained parameters

    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    parameters : dict
        Additional experiment parameters/hyperparameters

    Outputs to Disk
    ---------------
    Catalogue of class labels for each data sample.
    '''
    device = parameters['device']
    loadpath = parameters['saved_weights']
    savepath = os.path.dirname(loadpath)

    model.load_state_dict(torch.load(loadpath, map_location=device))
    model.eval()

    _, labels, _ = infer(dataloader, model, device)

    # pbar = tqdm(
    #     dataloader,
    #     leave=True,
    #     desc="Saving cluster labels",
    #     unit="batch",
    #     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    # )

    for batch in pbar:
        idx, batch = batch
        x = batch.to(device)
        _, labels, _ = infer(x)

        A = [{
            'idx': idx[i].cpu().detach().numpy(),
            'label': labels[i].cpu().detach().numpy()
        } for i in range(x.size(0))]

        utils.save_labels(
            [{k: v for k, v in d.items() if \
                (k == 'idx' or k == 'label')} for d in A],
            savepath
        )


def kmeans(model, dataloader, device):
    '''Initiate clusters using k-means algorithm.

    Parameters
    ----------
    model : PyTorch model instance

    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    device : PyTorch device object ('cpu' or 'gpu')

    Returns
    -------
    labels : array (M,)
        Sample-wise cluster assignment

    centroids : array (n_clusters,)
        Cluster centroids
    '''
    km = KMeans(
        n_clusters=model.n_clusters,
        max_iter=1000,
        n_init=100,
        random_state=2009
    )
    _, _, z_array = infer(dataloader, model, device)
    km.fit_predict(z_array)
    labels = km.labels_
    centroids = km.cluster_centers_
    return labels, centroids


def gmm(model, dataloader, device):
    '''Initiate clusters using Gaussian mixtures model algorithm.

    Parameters
    ----------
    model : PyTorch model instance

    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    device : PyTorch device object ('cpu' or 'gpu')

    Returns
    -------
    labels : array (M,)
        Sample-wise cluster assignment

    centroids : array (n_clusters,)
        Cluster centroids
    '''
    M = len(dataloader.dataset)
    labels, centroids = kmeans(model, dataloader, device)
    labels, counts = np.unique(labels, return_counts=True)
    gmm_weights = np.empty(len(labels))
    for i in range(len(labels)):
        gmm_weights[i] = counts[i] / M

    if device.type == 'cuda':
        cupy.cuda.Device(device.index).use()

    GMM = GaussianMixture(
        n_components=model.n_clusters,
        max_iter=1000,
        n_init=1,
        weights_init=gmm_weights,
        means_init=centroids
    )
    _, _, z_array = infer(dataloader, model, device)
    np.seterr(under='ignore')
    labels = GMM.fit_predict(z_array)
    centroids = GMM.means_
    return labels, centroids


def tsne(data):
    '''Perform t-SNE on data.

    Parameters
    ----------
    data : array (M,N)

    Returns
    -------
    results : array (M,2)
        2-D t-SNE embedding
    '''
    print('Running t-SNE...', end="", flush=True)
    M = len(data)
    np.seterr(under='warn')
    results = TSNE(
        n_components=2,
        perplexity=int(M/100),
        early_exaggeration=20,
        learning_rate=int(M/12),
        n_iter=2000,
        verbose=0,
        random_state=2009
    ).fit_transform(data.astype('float64'))
    print('complete.')
    return results


def infer(dataloader, model, device, v=False):
    '''Run DEC model in inference mode.

    Parameters
    ----------
    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    model : PyTorch model instance
        Model with trained parameters

    device : PyTorch device object ('cpu' or 'gpu')

    v : Boolean (default=False)
        Verbose mode

    Returns
    -------
    z_array : array (M,D)
        Latent space data (m_samples, d_features)
    '''
    if v:
        notqdm = False
    else:
        notqdm = True

    if hasattr(model, 'n_clusters'):
        cflag = True
    else:
        cflag = False
    model.eval()
    bsz = dataloader.batch_size
    z_array = np.zeros((len(dataloader.dataset), model.clustering.n_features), dtype=np.float32)

    if cflag:
        q_array = np.zeros((len(dataloader.dataset), model.n_clusters),dtype=np.float32)
        for b, batch in enumerate(tqdm(dataloader, disable=notqdm)):
        # for b, batch in enumerate(dataloader):
            _, batch = batch
            x = batch.to(device)
            q, _, z = model(x)
            q_array[b * bsz:(b*bsz) + x.size(0), :] = q.detach().cpu().numpy()
            z_array[b * bsz:(b*bsz) + x.size(0), :] = z.detach().cpu().numpy()
        labels = np.argmax(q_array.data, axis=1)
        return np.round(q_array, 5), labels, z_array
    else:
        for b, batch in enumerate(tqdm(dataloader, disable=notqdm)):
        # for b, batch in enumerate(dataloader):
            x = batch.to(device)
            _, z = model(x)
            z_array[b * bsz:(b*bsz) + x.size(0), :] = z.detach().cpu().numpy()
        return z_array


def target_distribution(q):
    '''From Xie/Girshick/Farhadi (2016). Computes the target distribution p,
    given soft assignements, q. The target distribtuion is generated by giving
    more weight to 'high confidence' samples - those with a higher probability
    of being a signed to a certain cluster.  This is used in the KL-divergence
    loss function.

    Parameters
    ----------
    q : array (M,D)
        Soft assignement probabilities - Probabilities of each sample being
        assigned to each cluster [n_samples, n_features]

    Returns
    -------
    p : array (M,D)
        Auxiliary target distribution of shape [n_samples, n_features].
    '''
    p = q ** 2 / np.sum(q, axis=0)
    p = np.transpose(np.transpose(p) / np.sum(p, axis=1))
    return np.round(p, 5)


def kmeans_metrics(dataloader, model, device, k_list):
    '''Run statistical evaluation on k-means over a range of cluster numbers.
    Calculates inertia, gap statistic (uniform and Gaussian), and silhouette
    score.

    Parameters
    ----------
    dataloader : PyTorch dataloader instance
        Loads data from disk into memory.

    model : PyTorch model instance
        Model with trained parameters

    device : PyTorch device object ('cpu' or 'gpu')

    k_list : array or list
        List of numbers of clusters to evaluate.

    Returns
    -------
    inertia : float
        k-means inertia

    silh : array
        k-means silhouette score

    gap_g : float
        k-means gap statistic (against uniform distribution)

    gap_u : float
        k-means gap statistic (against Gaussian distribution)
    '''
    _, _, z_array = infer(dataloader, model, device)

    feat_min = np.min(z_array, axis=0)
    feat_max = np.max(z_array, axis=0)
    feat_mean = np.mean(z_array, axis=0)
    feat_std = np.std(z_array, axis=0)

    gauss = np.zeros((z_array.shape[0], z_array.shape[1]))
    unifo = np.zeros((z_array.shape[0], z_array.shape[1]))

    for i in range(z_array.shape[1]):
        gauss[:,i] = np.random.normal(
            loc=feat_min[i],
            scale=feat_std[i],
            size=z_array.shape[0]
        )
        unifo[:,i] = np.random.uniform(
            low=feat_min[i],
            high=feat_max[i],
            size=z_array.shape[0]
        )

    inertia = np.zeros(len(k_list))
    inertiag = np.zeros(len(k_list))
    inertiau = np.zeros(len(k_list))
    silh = np.zeros(len(k_list))
    silhg = np.zeros(len(k_list))
    silhu = np.zeros(len(k_list))

    pbar = tqdm(
        k_list,
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        desc='Calculating k-means statistics'
    )

    for i, k in enumerate(pbar):
        complete = False
        attempt = 0
        while not complete:
            try:
                km = KMeans(n_clusters=k, n_init=100).fit(z_array)
                kmg = KMeans(n_clusters=k, n_init=100).fit(gauss)
                kmu = KMeans(n_clusters=k, n_init=100).fit(unifo)
                inertia[i] = km.inertia_
                inertiag[i] = kmg.inertia_
                inertiau[i] = kmu.inertia_
                silh[i] = silhouette_score(z_array, km.labels_)
                complete = True
                break
            except:
                if attempt == 5:
                    break
                complete = False
                attempt += 1
                continue

    gap_g = np.log(np.asarray(inertiag)) - np.log(np.asarray(inertia))
    gap_u = np.log(np.asarray(inertiau)) - np.log(np.asarray(inertia))
    return inertia, silh, gap_g, gap_u


# def plotter_mp(
#         fignames,
#         figpaths,
#         tb,
#         model,
#         dataloader,
#         device,
#         fname_dataset,
#         index_tra,
#         data_a,
#         data_b,
#         labels_a,
#         labels_b,
#         centroids_a,
#         centroids_b,
#         tsne_results,
#         epoch,
#         show,
#         latex=False
#     ):
#     '''Wrapper function for plotting DEC training and performance.
#
#     Parameters
#     ----------
#     fignames : list
#         List of figure names
#
#     figpaths : list
#         List of paths where to save figures
#
#     tb : Tensorboard SummaryWriter object
#
#     model : PyTorch model instance
#
#     dataloader : PyTorch dataloader instance
#         Loads data from disk into memory.
#
#     device : PyTorch device object ('cpu' or 'gpu')
#
#     fname_dataset : str
#         Path to dataset
#
#     index_tra: array
#         Indeces of data samples to be used for DEC training.
#
#     data_a : array (M,D)
#         Latent data from model initialization [m_samples,d_features]
#
#     data_b : array (M,D)
#         Latent data from current model state [m_samples,d_features]
#
#     labels_a : array (M,)
#         Class labels from cluster initialization
#
#     labels_b : array (M,)
#         Class labels from current state of clustering
#
#     centroids_a : array (n_clusters,)
#         Cluster centroids from model initialization
#
#     centroids_b : array (n_clusters,)
#         Current cluster centroids
#
#     tsne_results : array (M,2)
#         2-D t-SNE results from current model output
#
#     epoch : int
#         Current epoch of training
#
#     show : boolean
#         Show figures or not
#
#     latex : boolean (default=False)
#         Compile figures using latex (extremely slow - not recommended unless
#         rendering figures for publishing)
#
#     Outputs to Disk
#     ---------------
#     Figures analyzing DEC performance.
#     '''
#
#     figures = plotting.analyze_clustering(
#         model,
#         dataloader,
#         device,
#         fname_dataset,
#         index_tra,
#         data_a,
#         data_b,
#         labels_a,
#         labels_b,
#         centroids_a,
#         centroids_b,
#         tsne_results,
#         epoch,
#         show,
#         latex
#     )
#     [fig.savefig(f"{figpaths[i]}/{fignames[i]}_{epoch:03d}.png", dpi=300) \
#         for i, fig in enumerate(figures)]
#     [tb.add_figure(f"{fignames[i]}", fig, global_step=epoch, close=True) \
#         for i, fig in enumerate(figures)]
