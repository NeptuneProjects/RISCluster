from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime
import os
import random
import shutil
import sys
sys.path.insert(0, '../RISCluster/')

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
if sys.platform == 'darwin':
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
elif sys.platform == 'linux':
    from cuml import KMeans, TSNE
    # import cuml
    # from cuml import TSNE
    # from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

import importlib as imp
import plotting
imp.reload(plotting)
import utils
imp.reload(utils)

def pretrain(
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
        disp_idx,
        fname_dataset,
        savepath=savepath_run,
        show=show,
        mode='multi'
    )
    tb.add_figure('TrainingProgress', fig, global_step=0, close=True)

    if early_stopping:
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
                disp_idx,
                fname_dataset,
                savepath=savepath_run,
                show=show,
                mode='multi'
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

        if early_stopping:
            if epoch_val_mse < best_val_loss:
                strikes = 0
                best_val_loss = epoch_val_mse
                fname = f'{savepath_chkpnt}/AEC_Best_Weights.pt'
                torch.save(model.state_dict(), fname)
            else:
                strikes += 1

            if epoch > patience and strikes > patience:
                print('Stopping Early.')
                finished = True
                break
        else:
            fname = f'{savepath_chkpnt}/AEC_Params_{epoch:03d}}.pt'
            torch.save(model.state_dict(), fname)

    tb.add_hparams(
        {'Batch Size': batch_size, 'LR': lr},
        {
            'hp/Training MSE': epoch_tra_mse,
            'hp/Validation MSE': epoch_val_mse
        }
    )
    fig = plotting.compare_images(
        model,
        disp.to(device),
        epoch,
        disp_index,
        tvec,
        fvec,
        savepath_run,
        fname_dataset,
        show
    )
    tb.add_figure(
        'TrainingProgress',
        fig,
        global_step=epoch,
        close=True
    )
    fname = f'{savepath_run}/AEC_Params_Final.pt'
    if early_stopping and (finished == True or epoch == n_epochs-1):
        src_file = f'{savepath_chkpnt}AEC_Best_Weights.pt'
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
    return model, tb

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
    fname_dataset = parameters['fname_dataset']
    tbpid = parameters['tbpid']
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
        'Dashboard',
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
    # -- K-Means Initialization:
    print('Initiating clusters with k-means...', end="", flush=True)
    # labels_prev, centroids = kmeans(model, copy.deepcopy(dataloader), device)
    labels_prev, centroids = kmeans(model, dataloader, device)
    # -- GMM Initialization:
    # print('Initiating clusters with GMM...', end="", flush=True)
    # labels_prev, centroids = gmm(model, copy.deepcopy(dataloader), device)
    # -- K-Medoids Initialization:
    # print('Initiating clusters with k-medoids...', end="", flush=True)
    # labels_prev, centroids = kmeds(model, copy.deepcopy(dataloader), device)

    cluster_centers = torch.from_numpy(centroids).to(device)
    with torch.no_grad():
        model.state_dict()["clustering.weights"].copy_(cluster_centers)
    fname = f'{savepath_run}/DCM_Params_Initial.pt'
    torch.save(model.state_dict(), fname)
    print('complete.')
    # Initialize Target Distribution:
    q, _ = infer_labels(dataloader, model, device)
    z_array0 = infer_z(dataloader, model, device)
    p = target_distribution(q)
    epoch = 0
    figures = analyze_clustering(
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
        epoch,
        show
    )
    [fig.savefig(f"{figpaths[i]}/{fignames[i]}_{epoch:03d}.png", dpi=300) \
        for i, fig in enumerate(figures)]
    [tb.add_figure(f"{fignames[i]}", fig, global_step=epoch, close=True) \
        for i, fig in enumerate(figures)]

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
            # Uptade target distribution, check performance
            if (batch_num % update_interval == 0) and not \
                (batch_num == 0 and epoch == 0):
                q, labels = infer_labels(dataloader, model, device)
                p = target_distribution(q)
                # check stop criterion
                delta_label = np.sum(labels != labels_prev).astype(np.float32)\
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
            figures = analyze_clustering(
                model,
                dataloader,
                device,
                fname_dataset,
                index_tra,
                z_array0,
                infer_z(dataloader, model, device),
                labels_prev,
                labels,
                centroids,
                model.clustering.weights.detach().cpu().numpy(),
                epoch,
                show
            )
            [fig.savefig(f"{figpaths[i]}/{fignames[i]}_{epoch:03d}.png", dpi=300) \
                for i, fig in enumerate(figures)]
            [tb.add_figure(f"{fignames[i]}", fig, global_step=epoch, close=True) \
                for i, fig in enumerate(figures)]

        if finished:
            break

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

    fname = f'{savepath_run}/DCM_Params_Final.pt'
    torch.save(model.state_dict(), fname)
    tb.add_text("Path to Saved Weights", fname, global_step=None)
    tb.close()
    print('DCM parameters saved.')
    toc = datetime.now()
    print(f'Pre-training complete at {toc}; time elapsed = {toc-tic}.')
    return model

def predict(model, dataloader, parameters):
    device = parameters['device']
    loadpath = parameters['saved_weights']
    savepath = os.path.dirname(loadpath)

    model.load_state_dict(torch.load(loadpath, map_location=device))
    model.eval()

    pbar = tqdm(
        dataloader,
        leave=True,
        desc="Saving cluster labels",
        unit="batch",
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    for batch in pbar:
        idx, batch = batch
        batch.to(device)
        q, _, _ = model(x)
        label = torch.argmax(q, dim=1)

        A = [{
            'idx': idx[i].cpu().detach().numpy(),
            'label': label[i].cpu().detach().numpy()
        } for i in range(x.size(0))]

        utils.save_labels(
            [{k: v for k, v in d.items() if \
                (k == 'idx' or k == 'label')} for d in A],
            savepath
        )

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
    km = KMeans(
        n_clusters=model.n_clusters,
        max_iter=10000,
        n_init=500,
        random_state=2009
    )
    z_array = infer_z(dataloader, model, device)
    km.fit_predict(z_array)
    labels = km.labels_
    centroids = km.cluster_centers_
    return labels, centroids

def kmeds(model, dataloader, device):
    '''
    Initiate clusters using K-Medoids algorithm.
    # Arguments:
        model: PyTorch model instance
        dataloader: PyTorch dataloader instance
        device: PyTorch device object ('cpu' or 'gpu')
    # Inputs:
        n_clusters: Number of clusters, set during model construction.
        metric: l1 (choose from sk-learn metrics)
        max_iter: Max number of iterations for algorithm.
    # Returns:
        labels: Sample-wise cluster assignment
        centroids: Sample-wise cluster centroids
    '''
    kmed = KMedoids(
        n_clusters=model.n_clusters,
        metric='l1',
        init='heuristic',
        max_iter=10000,
        random_state=2009
    )
    z_array = infer_z(dataloader, model, device)
    kmed.fit_predict(z_array)
    labels = kmed.labels_
    centroids = kmed.cluster_centers_
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
    M = len(dataloader.dataset)
    labels, centroids = kmeans(model, dataloader, device)
    labels, counts = np.unique(labels, return_counts=True)
    gmm_weights = np.empty(len(labels))
    for i in range(len(labels)):
        gmm_weights[i] = counts[i] / M

    GMM = GaussianMixture(
        n_components=model.n_clusters,
        max_iter=10000,
        n_init=1,
        weights_init=gmm_weights,
        means_init=centroids
    )
    z_array = infer_z(dataloader, model, device)
    np.seterr(under='ignore')
    labels = GMM.fit_predict(z_array)
    centroids = GMM.means_
    return labels, centroids

def pca(labels, model, dataloader, device, tb, counter):
    z_array = infer_z(dataloader, model, device)
    row_max = z_array.max(axis=1)
    z_array /= row_max[:, np.newaxis]

    pca2 = PCA(n_components=model.n_clusters).fit(z_array)
    pca2d = pca2.transform(z_array)
    fig = plotting.view_clusters(pca2d, labels)
    tb.add_figure('PCA_Z', fig, global_step=counter, close=True)

def tsne(data, dataloader):
    print('Running t-SNE...', end="", flush=True)
    np.seterr(under='warn')
    results = TSNE(
        n_components=2,
        perplexity=int(len(dataloader.dataset)/100),
        early_exaggeration=20,
        learning_rate=int(len(dataloader.dataset)/12),
        n_iter=2000,
        verbose=0,
        random_state=2009
    ).fit_transform(data.astype('float64'))
    print('complete.')
    return results

def infer_labels(dataloader, model, device):
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
    model.eval()
    q_array = np.zeros(
        (len(dataloader.dataset), model.n_clusters),
        dtype=np.float32
    )
    bsz = dataloader.batch_size
    for b, batch in enumerate(dataloader):
        _, batch = batch
        x = batch.to(device)
        q, _, _ = model(x)
        q_array[b * bsz:(b*bsz) + x.size(0), :] = q.detach().cpu().numpy()
    labels = np.argmax(q_array.data, axis=1)
    return np.round(q_array, 5), labels

def infer_z(dataloader, model, device, v=False):
    if v:
        notqdm = False
    else:
        notqdm = True
    model.eval()
    z_array = np.zeros((len(dataloader.dataset), 10), dtype=np.float32)
    bsz = dataloader.batch_size
    for b, batch in enumerate(tqdm(dataloader, disable=notqdm)):
        _, batch = batch
        x = batch.to(device)
        if not hasattr(model, 'n_clusters'):
            _, z = model(x)
        else:
            _, _, z = model(x)
        z_array[b * bsz:(b*bsz) + x.size(0), :] = z.detach().cpu().numpy()
    return z_array

def target_distribution(q):
    '''
    From Xie/Girshick/Farhadi (2016). Computes the target distribution p, given
    soft assignements, q. The target distribtuion is generated by giving more
    weight to 'high confidence' samples - those with a higher probability of
    being a signed to a certain cluster.  This is used in the KL-divergence
    loss function.
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

def analyze_clustering(
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
        epoch,
        show=False
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
    fig1 = plotting.view_TSNE(tsne(data_b, dataloader), labels_b, title, show)
    fig2 = plotting.cluster_gallery(
        model,
        dataloader.dataset,
        fname_dataset,
        index_tra,
        device,
        data_b,
        labels_b,
        centroids_b,
        p,
        show
    )
    fig3 = plotting.centroid_dashboard(
        data_b,
        labels_b,
        centroids_b,
        n_clusters,
        p,
        show
    )
    fig4 = plotting.centroid_distances(
        data_b,
        labels_b,
        centroids_b,
        n_clusters,
        p,
        show
    )
    fig5 = plotting.view_latent_space(
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        n_clusters,
        p,
        show
    )
    fig6 = plotting.view_class_cdf(
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        n_clusters,
        p,
        show
    )
    fig7 = plotting.view_class_pdf(
        data_a,
        data_b,
        labels_a,
        labels_b,
        centroids_a,
        centroids_b,
        n_clusters,
        p,
        show
    )
    return [fig1, fig2, fig3, fig4, fig5, fig6, fig7]

def analyze_clustering2(
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
        epoch,
        show=False
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
    # title = f't-SNE Results - Epoch {epoch}'
    # fig1 = plotting.view_TSNE(tsne(data_b, dataloader), labels_b, title, show)
    fns = [
        plotting.cluster_gallery,
        plotting.centroid_dashboard,
        plotting.centroid_distances,
        plotting.view_latent_space,
        plotting.view_class_cdf,
        plotting.view_class_pdf
    ]
    A = [{
        'model': model,
        'dataset': dataloader.dataset,
        'fname_dataset': fname_dataset,
        'index_tra': index_tra,
        'device': device,
        'z_array': data_b,
        'labels': labels_b,
        'centroids': centroids_b,
        'p': p,
        'show': show
    }]
    A = A + [{
        'z_array': data_b,
        'labels': labels_b,
        'centroids': centroids_b,
        'n_clusters': n_clusters,
        'p': p,
        'show': show
    } for i in range(2)]
    A = A + [{
        'data_a': data_a,
        'data_b': data_b,
        'labels_a': labels_a,
        'labels_b': labels_b,
        'centroids_a': centroids_a,
        'centroids_b': centroids_b,
        'n_clusters': n_clusters,
        'p': p,
        'show': show
    } for i in range(3)]

    figures = []
    with ProcessPoolExecutor(max_workers=len(fns)) as exec:
        futures = [exec.submit(fns[i], **args) for i, args in enumerate(A)]
        for i, future in enumerate(as_completed(futures)):
            figures[i] = future.result()
    return figures

def kmeans_metrics(dataloader, model, device, k_list):
    z_array = infer_z(dataloader, model, device)

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
                if sys.platform == 'darwin':
                    print('Using CPU to calculate.')
                    km = KMeans(n_clusters=k, n_init=100).fit(z_array)
                    kmg = KMeans(n_clusters=k, n_init=100).fit(gauss)
                    kmu = KMeans(n_clusters=k, n_init=100).fit(unifo)
                elif sys.platform == 'linux':
                    km = cuml.KMeans(n_clusters=k, n_init=100).fit(z_array)
                    kmg = cuml.KMeans(n_clusters=k, n_init=100).fit(gauss)
                    kmu = cuml.KMeans(n_clusters=k, n_init=100).fit(unifo)
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

def measure_class_inertia(data, centroids, n_clusters):
    inertia = np.empty(n_clusters)
    for j in range(n_clusters):
        mu = centroids[j]
        inertia[j] = np.sum(np.sqrt(np.sum((data - mu) ** 2, axis=1)) ** 2)
    return inertia
