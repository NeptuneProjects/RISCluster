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
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

import importlib as imp
import plotting
imp.reload(plotting)
import utils
imp.reload(utils)

def pretrain_DCEC(
        model,
        dataloaders,
        criteria,
        optimizer,
        batch_size,
        lr,
        parameters
    ):
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

    training_history = {'mse': [], 'mae': []}
    validation_history = {'mse': [], 'mae': []}

    tra_loader = dataloaders[0]
    val_loader = dataloaders[1]
    M_tra = len(tra_loader.dataset)
    M_val = len(val_loader.dataset)

    images = next(iter(tra_loader))
    grid = torchvision.utils.make_grid(images)

    disp_idx = sorted(np.random.randint(0, images.size(0), 4))
    disp = images[disp_idx]

    tb = SummaryWriter(log_dir=savepath_run)
    tb.add_image('images', grid)
    tb.add_graph(model, images)

    finished = False
    for epoch in range(n_epochs):
        # ==== Training Loop: =================================================
        model.train(True)

        running_tra_mse = 0.0
        running_tra_mae = 0.0

        for batch in tra_loader:
            x = batch.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                x_rec = model(x)
                loss_mse = criterion_mse(x_rec, x)
                loss_mae = criterion_mae(x_rec, x)
                loss_mse.backward()
                optimizer.step()

            running_tra_mse += loss_mse * x.size(0)
            running_tra_mae += loss_mae * x.size(0)

        epoch_tra_mse = running_tra_mse / M_tra
        epoch_tra_mae = running_tra_mae / M_tra
        training_history['mse'].append(epoch_tra_mse.cpu().detach().numpy())
        training_history['mae'].append(epoch_tra_mae.cpu().detach().numpy())
        tb.add_scalar('Training MSE', epoch_tra_mse, epoch)
        tb.add_scalar('Training MAE', epoch_tra_mae, epoch)

        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        if epoch == 0 or (epoch % 5) == 0:
            plotting.compare_images(
                model,
                tra_loader,
                disp,
                epoch,
                savepath_run,
                show
            )
        # ==== Validation Loop: ===============================================
        model.train(False)

        running_val_mse = 0.0
        running_val_mae = 0.0

        if early_stopping:
            savepath_chkpnt = f'{savepath_run}tmp/'
            if not os.path.exists(savepath_chkpnt):
                os.makedirs(savepath_chkpnt)
            val_loss = 0.0
            best_val_loss = 10000
            running_size = 0

        for batch in val_loader:
            model.eval()
            with torch.no_grad():
                x = batch.to(device)
                x_rec = model(x)
                loss_mse = criterion_mse(x_rec, x)
                loss_mae = criterion_mae(x_rec, x)

            running_val_mse += loss_mse * x.size(0)
            running_val_mae += loss_mae * x.size(0)

            if early_stopping:
                running_size += x.size(0)
                val_loss = running_val_mse / running_size

                if val_loss < best_val_loss:
                    strikes = 0
                    best_val_loss = val_loss
                    fname = f'{savepath_chkpnt}AEC_Best_Weights.pt'
                    torch.save(model.state_dict(), fname)
                else:
                    strikes += 1

                if epoch > patience and strikes > patience:
                    print('Stopping Early.')
                    finished = True
                    break

        if finished:
            M_val = running_size
        epoch_val_mse = running_val_mse / M_val
        epoch_val_mae = running_val_mae / M_val
        validation_history['mse'].append(epoch_val_mse.cpu().detach().numpy())
        validation_history['mae'].append(epoch_val_mae.cpu().detach().numpy())
        tb.add_scalar('Validation MSE', epoch_val_mse, epoch)
        tb.add_scalar('Validation MAE', epoch_val_mae, epoch)

        print(
            f'Epoch [{epoch+1}/{n_epochs}] | Training: '
            f'MSE = {epoch_tra_mse:.4f}, MAE = {epoch_tra_mae:.4f} | '
            f'Validation: MSE = {epoch_val_mse:.4f}, '
            f'MAE = {epoch_val_mae:.4f}'
        )

        if finished:
            break

    if early_stopping and (finished == True or epoch == n_epochs-1):
        src_file = f'{savepath_chkpnt}AEC_Best_Weights.pt'
        dst_file = f'{savepath_run}AEC_Params_{serial_run}.pt'
        shutil.move(src_file, dst_file)
    else:
        fname = f'{savepath_run}AEC_Params_ {serial_run}.pt'
        torch.save(model.state_dict(), fname)
    print('AEC parameters saved.')

    utils.save_history(
        training_history,
        validation_history,
        savepath_run,
        serial_run
        )
    tb.close()
    toc = datetime.now()
    print(f'Pre-training complete at {toc}; time elapsed = {toc-tic}.')
    return model

def train_DCEC(
        model,
        dataloader,
        criteria,
        optimizer,
        batch_size,
        lr,
        gamma,
        tol,
        parameters
        ):
    '''
    Function facilitates training (i.e., clustering and fine-tuning of AEC) and
    of the DCEC model.
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
    n_clusters = parameters['n_clusters']
    update_interval = parameters['update_interval']
    savepath_exp = parameters['savepath']
    show = parameters['show']
    mode = parameters['mode']
    loadpath = parameters['saved_weights']

    model.load_state_dict(
        torch.load(loadpath, map_location=device), strict=False
    )

    savepath_run, serial_run = utils.init_output_env(
        savepath_exp,
        mode,
        **{
        'batch_size': batch_size,
        'lr': lr,
        'gamma': gamma,
        'tol': tol
        }
    )

    criterion_mse = criteria[0]
    criterion_kld = criteria[1]

    training_history = {'iter': [], 'mse': [], 'kld': [], 'loss': []}

    M = len(dataloader.dataset)

    images = next(iter(dataloader))
    grid = torchvision.utils.make_grid(images)

    tb = SummaryWriter(log_dir = savepath_run)
    tb.add_image('images', grid)
    tb.add_graph(model, images)

    # Initialize Clusters:
    kmeans(model, copy.deepcopy(dataloader), device)
    # Initialize Target Distribution:
    q, preds_prev = predict_labels(model, dataloader, device)
    p = target_distribution(q)

    total_counter = 0
    finished = False
    for epoch in range(n_epochs):
        model.train(True)

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0
        running_size = 0
        # batch_num = 0
        # Iterate over data:
        for batch_num, batch in enumerate(dataloader):
            x = batch.to(device)
            # Uptade target distribution, check performance
            if (batch_num % update_interval == 0) and not \
                (batch_num == 0 and epoch == 0):
                q, preds = predict_labels(model, dataloader, device)
                p = target_distribution(q)
                # check stop criterion
                delta_label = np.sum(preds != preds_prev).astype(np.float32) \
                              / preds.shape[0]
                preds_prev = np.copy(preds)
                if delta_label < tol:
                    print('Stop criterion met, training complete.')
                    finished = True
                    break

            tar_dist = p[running_size:(running_size + x.size(0)), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                q, x_rec, _ = model(x)
                loss_rec = criterion_mse(x_rec, x)
                loss_clust = criterion_kld(torch.log(q), tar_dist) / x.size(0)
                loss = loss_rec + gamma * loss_clust
                loss.backward()
                optimizer.step()

            running_size += x.size(0)
            running_loss += loss * x.size(0)
            running_loss_rec += loss_rec * x.size(0)
            running_loss_clust += loss_clust * x.size(0)

            accum_loss = running_loss / running_size
            accum_loss_rec = running_loss_rec / running_size
            accum_loss_clust = running_loss_clust / running_size

            training_history['iter'].append(total_counter)
            training_history['loss'].append(
                accum_loss.cpu().detach().numpy()
            )
            training_history['mse'].append(
                accum_loss_rec.cpu().detach().numpy()
            )
            training_history['kld'].append(
                accum_loss_clust.cpu().detach().numpy()
            )

            tb.add_scalar('Loss', accum_loss, total_counter)
            tb.add_scalar('MSE', accum_loss_rec, total_counter)
            tb.add_scalar('KLD', accum_loss_clust, total_counter)

            for name, weight in model.named_parameters():
                tb.add_histogram(name, weight, total_counter)
                tb.add_histogram(f'{name}.grad', weight.grad, total_counter)

            print(
                f'Epoch [{epoch+1}/{n_epochs}] Batch [{batch_num}]| '
                f'Training: Loss = {accum_loss:.9f}, '
                f'MSE = {accum_loss_rec:.9f}, KLD = {accum_loss_clust:.9f}'
            )

            batch_num += 1
            total_counter += 1

        if finished:
            break

    fname = f'{savepath_run}DCEC_Params_{serial_run}.pt'
    torch.save(model.state_dict(), fname)
    print('DCEC parameters saved.')

    utils.save_history(training_history, None, savepath_run, serial_run)
    tb.close()
    toc = datetime.now()
    print(f'Pre-training complete at {toc}; time elapsed = {toc-tic}.')
    return model

def predict_DCEC(model, dataloader, idx_smpl, parameters):
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

    label_list = []

    running_size = 0
    counter = 0
    for batch_num, batch in enumerate(dataloader):
        x = batch.to(device)
        q, x_rec, z = model(x)
        label = torch.argmax(q, dim=1)
        A = [{
            'x': x[i].cpu().detach().numpy(),
            'label': label[i].cpu().detach().numpy(),
            'x_rec': x_rec[i].cpu().detach().numpy(),
            'z': z[i].cpu().detach().numpy(),
            'idx': idx_smpl[running_size:(running_size + x.size(0))][i],
            'savepath': savepath_run[int(label[i])]} for i in range(x.size(0))]
        print('--------------------------------------------------------------')
        print(f'Saving outputs for Batch {batch_num}:')
        utils.save_labels(
            [{k: v for k, v in d.items() if \
                (k == 'idx' or k == 'label')} for d in A],
            savepath_exp,
            serial_exp
        )
        print('Saving spectrograms to file...')
        # Parallel Implementation
        with ProcessPoolExecutor(max_workers=max_workers) as exec:
            futures = [exec.submit(plotting.save_DCEC_output, **a) for a in A]
            kwargs = {
                'total': len(futures),
                'unit': 'it',
                'unit_scale': True,
                'leave': True
            }
            for future in tqdm(as_completed(futures), **kwargs):
                future.result()
        # Serial Implementation:
        # for i in tqdm(range(x.size(0))):
        #     plotting.save_DCEC_output(
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
        weights: Assigned to model's clustering layer weights
    '''
    km = KMeans(n_clusters=model.n_clusters, n_init=20)
    z_array = None
    model.eval()
    for batch in dataloader:
        x = batch.to(device)
        _, _, z = model(x)
        if z_array is not None:
            z_array = np.concatenate((z_array, z.cpu().detach().numpy()), 0)
        else:
            z_array = z.cpu().detach().numpy()

    # Perform K-means
    km.fit_predict(z_array)
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_weight(weights.to(device))

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
        preds [n_samples,]: Hard assigned label based on max of q_array
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

    preds = np.argmax(q_array.data, axis=1)
    return np.round(q_array, 5), preds

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
