
from contextlib import redirect_stdout
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, '../../RISCluster/')

import h5py
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from RISCluster.utils.utils import notify

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2048, 32),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.latent2dec = nn.Sequential(
            nn.Linear(32, 2048),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        x = self.latent2dec(x)
        x = x.view(-1, 64, 4, 8)
        x = self.decoder(x)
        return x

class AEC(nn.Module):
    def __init__(self, encoder, decoder):
        super(AEC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number,
        feature_dim,
        alpha = 1.0,
        cluster_centers = None
    ):
        super(ClusterAssignment, self).__init__()
        self.feature_dim = feature_dim
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.feature_dim, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, x: torch.Tensor):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        feature_dim: int,
        encoder,
        alpha
    ):
        super(DEC, self).__init__()
        self.encoder = encoder
        self.cluster_number = cluster_number
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            self.cluster_number, self.feature_dim, self.alpha
        )

    def forward(self, x):
        return self.assignment(self.encoder(x))

class DCEC(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            feature_dim: int,
            autoencoder,
            alpha
    ):
        super(DCEC, self).__init__()
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.cluster_number = cluster_number
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            self.cluster_number, self.feature_dim, self.alpha
        )

    def forward(self, x):
        z = self.encoder(x)
        q = self.assignment(z)
        x = self.decoder(z)
        return q, x

def target_distribution(q):
    weight = (q ** 2) / torch.sum(q, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def train():
    '''Wrapper function for training the DEC.'''
    pass

def predict():
    '''Wrapper function for running the DEC.'''
    pass

def pretrain(
        train_loader,
        val_loader,
        run_serial,
        epochs: int,
        batch_size: int,
        LR=0.0001,
        show=True,
        send_message=True,
        savepath='.'
    ):
    '''Wrapper function for training the autoencoder.'''
    # Define the training step to be executed every iteration:
    def pretraining_step(engine, batch):
        autoencoder.train()
        optimizer.zero_grad()
        x = batch.to(device)
        x_pred = autoencoder(x)
        loss = criterion_mse(x_pred, x)
        loss.backward()
        mae = criterion_mae(x_pred, x)
        optimizer.step()
        return loss.item(), mae.item()
    # Instantiate trainer engine using the pre-defined training step function.
    trainer = Engine(pretraining_step)
    # Define the validation step to be executed every iteration:
    def validation_step(engine, batch):
        autoencoder.eval()
        with torch.no_grad():
            x = batch.to(device)
            x_pred = autoencoder(x)
            return x_pred, x

    evaluator = Engine(validation_step)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA device available, using GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA device not available, using CPU.')

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    autoencoder = AEC(encoder, decoder).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=LR)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    pretraining_history = {'mse': [], 'mae': []}
    validation_history = {'mse': [], 'mae': []}

    MeanSquaredError(device=device).attach(evaluator, 'mse')
    MeanAbsoluteError(device=device).attach(evaluator, 'mae')

    def print_pretraining_log(engine, dataloader, mode, history_dict):
        evaluator.run(dataloader)
        metrics = evaluator.state.metrics
        if mode == 'Training':
            print(
                f'Epoch[{trainer.state.epoch}/{epochs}] | Training Results: '
                f'MSE = {metrics["mse"]:.2f}, MAE = {metrics["mae"]:.2f} | ',
                end='', flush='True'
            )
        elif mode == 'Validation':
            print(
                f'Validation Results: MSE = {metrics["mse"]:.2f}, '
                f'MAE = {metrics["mae"]:.2f}'
            )
        else:
            raise ValueError('Incorrect evaluation mode input: Choose'
                             '"Training" or "Validation"')

        for key in evaluator.state.metrics.keys():
            history_dict[key].append(evaluator.state.metrics[key])

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        print_pretraining_log,
        train_loader,
        'Training',
        pretraining_history
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        print_pretraining_log,
        val_loader,
        'Validation',
        validation_history
    )

    def compare_images(engine, disp, show=show, savepath=savepath):
        epoch = engine.state.epoch
        reconstructed_images = autoencoder(disp)
        figtitle = 'AEC Training Run {}: Epoch {}'.format(run_serial,epoch)
        n, o = list(disp.size()[2:])
        fig = view_specgram_training(
            disp,
            reconstructed_images,
            n, o,
            figtitle,
            figsize=(12,9),
            show=show
        )
        savepath_snap = savepath + f'Snapshots{run_serial}/'
        figname = savepath_snap + f'AEC_Training_Epoch_{epoch:03d}.png'
        fig.savefig(figname)

    disp = next(iter(train_loader))
    disp_dim = list(disp.size())[0]
    disp_idx = sorted(np.random.randint(0, disp_dim, 4))
    disp = disp[disp_idx]
    trainer.add_event_handler(
        Events.STARTED,
        compare_images,
        disp.to(device),
        show,
        savepath=savepath
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=5),
        compare_images,
        disp.to(device),
        show,
        savepath=savepath
    )

    def score_function(engine):
        mse = engine.state.metrics['mse']
        return -mse

    early_stop_handler = EarlyStopping(
        patience=10,
        score_function=score_function,
        trainer=trainer
    )
    evaluator.add_event_handler(
        Events.COMPLETED,
        early_stop_handler
    )

    checkpointer = ModelCheckpoint(
        savepath + '/tmp',
        'best',
        score_function=score_function
    )
    evaluator.add_event_handler(
        Events.COMPLETED,
        checkpointer,
        {'autoencoder': autoencoder}
    )

    tic=datetime.now()
    print('--------------------------------------------------------------')
    print('Commencing AEC run {} at {}'.format(run_serial, tic))
    trainer.run(train_loader, max_epochs=epochs)
    toc = datetime.now()

    if send_message:
        msgsubj = 'ConvAEC Training Complete'
        msgcontent = f'''ConvAEC training completed at {toc}.
        Time Elapsed = {(toc-tic)}.'''
        notify(msgsubj, msgcontent)

    fname = savepath + 'AEC_Params_' + run_serial + '.pt'
    torch.save(autoencoder.state_dict(),fname)
    print('AEC parameters saved.')

    save_history(pretraining_history, validation_history, savepath, run_serial)
    try:
        fig = view_learningcurve(
            pretraining_history,
            validation_history,
            show=show
        )
        fname = savepath + 'AEC_LossCurve_' + run_serial
        fig.savefig(fname)
        print('Loss curves saved.')
    except:
        plt.close()
        print('Unable to save loss curves.')

    print(f'Elapsed Time: {toc - tic}')
    print('--------------------------------------------------------------')
    return autoencoder, pretraining_history, validation_history

# class ClusteringLayer(tf.keras.layers.Layer):
#     """
#     Clustering layer converts input sample (feature) to soft label, i.e. a
#     vector that represents the probability of the sample belonging to each
#     cluster. The probability is calculated with student's t-distribution.
#     # Example
#     ```
#         model.add(ClusteringLayer(n_clusters=10))
#     ```
#     # Arguments
#         n_clusters: number of clusters.
#         weights: list of Numpy array with shape `(n_clusters, n_features)`
#         which represents the initial cluster centers.
#         alpha: parameter in Student's t-distribution. Default to 1.0.
#     # Input shape
#         2D tensor with shape: `(n_samples, n_features)`.
#     # Output shape
#         2D tensor with shape: `(n_samples, n_clusters)`.
#     """
#     def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
#         super(ClusteringLayer, self).__init__(**kwargs)
#         #initialize object attributes
#         self.n_clusters = n_clusters
#         self.alpha = alpha #exponent for soft assignment calculation
#         self.initial_weights = weights
#         self.input_spec = tf.keras.layers.InputSpec(ndim=2)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 2
#         input_dim = input_shape[1]
#         self.input_spec = tf.keras.layers.InputSpec(dtype=tfkb.floatx(),
#                                                     shape=(None, input_dim))
#         self.clusters = self.add_weight(name='clusters',
#                                         shape=(self.n_clusters,
#                                         int(input_dim)),
#                                         initializer='glorot_uniform')
#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#         self.built = True
#
#     def call(self, inputs, **kwargs):
#         q = 1.0 / (1.0 + (tfkb.sum(tfkb.square(tfkb.expand_dims(inputs,
#                                    axis=1) - self.clusters), axis=2) \
#                                    / self.alpha))
#         q **= (self.alpha + 1.0) / 2.0
#         q = tfkb.transpose(tfkb.transpose(q) / tfkb.sum(q, axis=1))
#         return q
#
#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) == 2
#         return input_shape[0], self.n_clusters
#
#     def get_config(self):
#         config = {'n_clusters': self.n_clusters}
#         base_config = super(ClusteringLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
# class ConvAEC:
#     def build(m, n, o, p, depth, strides, activation, kernel_init, latent_dim):
#
#         # Define Input Dimensions:
#         img_input = tf.keras.layers.Input(shape=(n,o,p))
#
#         # Define the encoder:
#         e = tf.keras.layers.Conv2D(depth*2**0, (5,5), strides = strides,
#                         activation=activation, kernel_initializer=kernel_init,
#                         padding='same')(img_input)
#         e = tf.keras.layers.Conv2D(depth*2**1, (5,5), strides = strides,
#                         activation=activation, kernel_initializer=kernel_init,
#                         padding='same')(e)
#         e = tf.keras.layers.Conv2D(depth*2**2, (3,3), strides = strides,
#                         activation=activation, kernel_initializer=kernel_init,
#                         padding='same')(e)
#         e = tf.keras.layers.Conv2D(depth*2**3, (3,3), strides = strides,
#                         activation=activation, kernel_initializer=kernel_init,
#                         padding='same')(e)
#         shape_before_flattening = tf.keras.backend.int_shape(e)
#         e = tf.keras.layers.Flatten()(e)
#         encoded = tf.keras.layers.Dense(latent_dim, activation=activation,
#                                         name='encoded')(e)
#
#         # Define the decoder:
#         d = tf.keras.layers.Dense(np.prod(shape_before_flattening[1:]),
#                                   activation='relu')(encoded)
#         d = tf.keras.layers.Reshape(shape_before_flattening[1:])(d)
#         d = tf.keras.layers.Conv2DTranspose(depth*2**2, (3,3), strides=strides,
#                         activation=activation, kernel_initializer=kernel_init,
#                         padding='same')(d)
#         d = tf.keras.layers.Conv2DTranspose(depth*2**1, (5,5), strides=strides,
#                         activation=activation, kernel_initializer=kernel_init,
#                         padding='same')(d)
#         d = tf.keras.layers.Conv2DTranspose(depth*2**0, (5,5), strides=strides,
#                         activation=activation, kernel_initializer=kernel_init,
#                         padding='same')(d)
#         decoded = tf.keras.layers.Conv2DTranspose(1, (5,5), strides=strides,
#                         activation='linear', kernel_initializer=kernel_init,
#                         padding='same')(d)
#
#         # Create the models (Encoder, Autoencoder):
#         encoder = tf.keras.models.Model(inputs=img_input, outputs=encoded,
#                                         name='Encoder')
#         autoencoder = tf.keras.models.Model(inputs=img_input, outputs=decoded,
#                                             name='Autoencoder')
#         return encoder, autoencoder
#
# def build_dtvec(starttime, dt):
#     pass
#     return None
#
# def build_fvec(cutoff, fs):
#     pass
#     return None

def get_metadata(query_index, sample_index, fname_dataset):
    '''Returns station metadata given sample index.'''
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/30sec/Catalogue'
        dset = f[DataSpec]
        metadata = dict()
        counter = 0
        for i in query_index:
            query = sample_index[i]
            metadata[counter] = json.loads(dset[query])
            counter += 1
    return metadata

# def get_trace():
#     pass
#     return None
#
# def init_GPU(GPU_frac=0.5):
#     '''Designate GPUs and limit memory.'''
#     os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"]="0"
#     GPU_options = tfcv1.GPUOptions(per_process_gpu_memory_fraction=GPU_frac,
#                                    allow_growth=True)
#     sess = tfcv1.Session(config=tfcv1.ConfigProto(gpu_options=GPU_options))
#     tfcv1.keras.backend.set_session(sess)

def init_aec_output_env():
    run_serial = datetime.now().strftime('%Y%m%dT%H%M%S')
    savepath = '../../../Outputs/Models/AEC/'
    savepath_run = savepath + 'Run' + run_serial + '/'
    savepath_chkpnt = savepath_run + 'tmp/'
    savepath_snap = savepath_run + f'Snapshots{run_serial}/'

    folders = [
        savepath_run,
        savepath_chkpnt,
        savepath_snap
    ]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    print('New AEC run file structure created at:\n'
          f'{savepath_run}')
    return savepath_run, savepath_chkpnt, savepath_snap, run_serial

# def load_test(fname_dataset, M, index_test):
#     with h5py.File(fname_dataset, 'r') as f:
#         #samples, frequency bins, time bins, amplitude
#         DataSpec = '/30sec/Spectrogram'
#         dset = f[DataSpec]
#         m, n, o = dset.shape
#         print(f'H5 file has {m} samples, {n} frequency bins, {o} time bins.')
#         print(f'Loading {M} test samples...')
#         tic = datetime.now()
#
#         np.seterr(divide='raise')
#         X = np.empty([M, n-2, o-173, 1])
#         sample_index = np.empty([M,], dtype=np.int)
#         dset_arr = np.empty([1, n, o])
#         count = 0
#         for i in range(M):
#             try:
#                 dset_arr = dset[index_test[i], 1:-1, 1:129]
#                 dset_arr /= dset_arr.max()
#                 X[count,:,:,:] = dset_arr[..., np.newaxis]
#                 sample_index[count,] = int(index_test[i])
#                 count += 1
#             except:
#                 print('Numpy "Divide-by-zero Warning" raised, '
#                       'skipping spectrogram.')
#                 pass
#
#             print('%.2f' % (float(100*i/(M-1))) + '% complete.', end='\r')
#         toc = datetime.now()
#         print(f'\nTime elapsed = {toc}')
#
#     # Update dimensions of X:
#     m, n, o, p = X.shape
#     print(f'Shape of X is {(m, n, o, p)}')
#     msgsubj = 'Training/Validation Data Loaded'
#     msgcontent = f'''{M} training/validation spectrograms loaded successfully.
# Time Elapsed = {(toc-tic)}'''
#     notify(msgsubj, msgcontent)
#     return X, m, n, o, p, sample_index
#

def load_autoencoder(fname, device):
    encoder = Encoder()
    decoder = Decoder()
    autoencoder = AEC(encoder, decoder)
    autoencoder.load_state_dict(torch.load(fname, map_location=device))
    autoencoder.eval()
    print('Encoder, Decoder, and Autoencoder loaded.')
    return encoder, decoder, autoencoder

def load_data(fname_dataset, M, index, send_message=True):
    '''
    *M* random spectrograms are read in and pre-processed iteratively as
    follows:
    1. Open the file and specified h5 dataset.
    2. Read the *m*th spectrogram from file.  Each spectrogram is of shape
    (*n*, *o*).
    3. Omit unnecessary data when assigning data into memory to cut down on
    memory usage and to make subsequent convolutions and transpositions
    straightforward with regard to data dimensions.  Specifically, index *n*[0]
    contains low frequency data that is of no interest for our high-frequency
    analaysis; *n*[65] is a vector of time values; and *o*[0] is a vector of
    frequency values.  Additional *o* indices may be omitted to reduce the
    length of spectrogram being analyzed.
    4. Add a fourth axis for the amplitude dimension.  This is requried for
    subsequent steps.
    At the conclusion of reading in and pre-processing, data has been omitted
    in such a way that the final shape of the array is
    (*m*, *n*, *o*, *p*) = (*M*, 64, 128, 1).
    Note: There are two other methods to accomplish the read.  The first is to
    use fancy indexing by referring to the array of random indices in a single
    line (i.e., arr = dset[rand_indices,:,:,:]).  A similar method is to use
    the "read_direct" function.  However, these two methods take at least an
    order of magnitude longer to read in the data than to iterate over each
    index.  The long duration is a result of how the .h5 file is written to
    disk.  In our case, the spectrogram data has been written in chunks that
    are optimized for reading into this workflow.
    '''
    with h5py.File(fname_dataset, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/30sec/Spectrogram'
        dset = f[DataSpec]
        m, n, o = dset.shape
        print('--------------------------------------------------------------')
        print(f'H5 file has {m} samples, {n} frequency bins, {o} time bins.')
        print(f'Loading {M} samples...')
        tic = datetime.now()

        np.seterr(all='raise')
#         X = np.empty([M, n-2, o-173, 1])
        X = torch.empty([M, 1, n-2, o-173])
        idx_sample = np.empty([M,], dtype=np.int)
        dset_arr = np.empty([n, o])
#         dset_arr = torch.empty([1, n, o])
        count = 0
        for i in range(M):
            try:
                dset_arr = dset[index[i], 1:-1, 1:129]
                dset_arr /= dset_arr.max()
                X[count,:,:,:] = torch.from_numpy(np.expand_dims(dset_arr,
                                                                 axis=0))
                idx_sample[count,] = int(index[i])
                count += 1
            except:
                print('Numpy "Divide-by-zero Warning" raised, '
                      'skipping spectrogram.')
                print('Sample Index = {}'.format(index[i]))
                print(dset[index[i], 1:-1, 1:129])
                pass

            print('    %.2f' % (float(100*i/(M-1))) + '% complete.', end='\r')
        toc = datetime.now()
        print(f'\nTime elapsed = {toc-tic}')

    m, p, n, o = list(X.size())
    print(f'Shape of output is {(m, p, n, o)}')
    if send_message:
        msgsubj = 'Data Loaded'
        msgcontent = f'''{M} spectrograms loaded successfully at {toc}.
Time Elapsed = {(toc-tic)}'''
        notify(msgsubj, msgcontent)
    print('--------------------------------------------------------------')
    return X, m, p, n, o, idx_sample

# def optim_and_cluster(X_train, model, batch_size, tol, maxiter,
#                       update_interval, labels, labels_last):
#     loss = 0              # initialize loss
#     index = 0             # initialize index to start
#     tic = datetime.now()
#     loss_list = np.zeros([maxiter,3])                       # Keep track of loss function during training process
#     index_array = np.arange(X_train.shape[0])
#     for ite in range(int(maxiter)):
#         if ite % update_interval == 0:
#             q, reconst  = model.predict(X_train, verbose=1) # Calculate soft assignment distribtuion & CAE reconstructions
#             p = target_distribution(q)                      # Update the auxiliary target distribution p
#             labels = q.argmax(1)                            # Assign labels to the embedded latent space samples
#             # check stop criterion - Calculate the % of labels that changed from previous update
#             delta_label = np.sum(labels != labels_last).astype(np.float32) /labels.shape[0]
#             labels_last = np.copy(labels)                   # Generate copy of labels for future updates
#             loss= np.round(loss, 5)                         # Round the loss
#
#             print('Iter %d' % ite)
#             print('Loss: {}'.format(loss))
#             print_cluster_size(labels)                      # Show the number of samples assigned to each cluster
#
#             if ite > 0 and delta_label < tol:               # Break training if loss reaches the tolerance threshhold
#                 print('delta_label ', delta_label, '< tol ', tol)
#                 break
#
#         idx = index_array[index * batch_size: min((index+1) * batch_size, X_train.shape[0])]
#         loss = model.train_on_batch(x=X_train[idx], y=[p[idx], X_train[idx,:,:,:]])
#         index = index + 1 if (index + 1) * batch_size < X_train.shape[0] else 0
#     toc = datetime.now()
#     print(f'Deep Embedded Clustering Computation Time: {toc-tic}')
#     msgsubj = 'DEC Training Complete'
#     msgcontent = f'''DEC training completed at {datetime.now()}.
# Time Elapsed = {toc-tic}.'''
#     notify(msgsubj, msgcontent)
#     return model, reconst
#
# def print_cluster_size(labels):
#     """
#     Shows the number of samples assigned to each cluster.
#     # Example
#     ```
#         print_cluster_size(labels=kmeans_labels)
#     ```
#     # Arguments
#         labels: 1D array  of clustering assignments. The value of each label corresponds to the cluster
#                 that the samples in the clustered data set (with the same index) was assigned to. Array must be the same length as
#                 data.shape[0]. where 'data' is the clustered data set.
#     """
#     num_labels = max(labels) + 1
#     for j in range(0,num_labels):
#         label_idx = tf.where(labels==j)[0]
#         print("Label " + str(j) + ": " + str(label_idx.shape[0]))
#
# def save_DEC_labelInds(fname, n_clusters, labels):
#     with h5py.File(fname, 'w') as nf:
#         for i in range(0, n_clusters):
#             fname = 'label_{0}_indices'.format(i)
#             label_idx = np.where(labels==i)[0]
#             nf.create_dataset(fname, data=label_idx, dtype=label_idx.dtype)
#
# def save_DEC_lspace(fname, enc_test):
#     with h5py.File(fname, 'w') as nf:
#         nf.create_dataset('EncodedData', data=enc_test, dtype=enc_test.dtype)
#
# def save_model_info(model, fname):
#     with open(fname + '.txt', 'w') as f:
#         with redirect_stdout(f):
#             model.summary()
#     tf.keras.utils.plot_model(model, to_file=fname + '.png', show_shapes=True)
#
# def save_trained_lspace(fname, train_enc, val_enc, val_reconst):
#     with h5py.File(fname, 'w') as nf:
#         nf.create_dataset('Train_EncodedData', data=train_enc,
#                           dtype=train_enc.dtype)
#         nf.create_dataset('Val_EncodedData', data=val_enc,
#                           dtype=val_enc.dtype)
#         nf.create_dataset('Val_Reconst', data=val_reconst,
#                           dtype=val_reconst.dtype)
#

def save_history(training_history, validation_history, savepath, run_serial):
    fname = savepath + 'AEC_History{}.csv'.format(run_serial)
    d1 = training_history.copy()
    d2 = validation_history.copy()
    modes = ['training', 'validation']
    for mode in range(len(modes)):
        if modes[mode] == 'training':
            d = d1
        elif modes[mode] == 'validation':
            d = d2
        for j in range(2):
            newkey = '{}_{}'.format(modes[mode], list(d.keys())[0])
            oldkey = list(d.keys())[0]
            d[newkey] = d.pop(oldkey)
    d2.update(d1)
    del d1

    with open(fname, 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(d2.keys())
        w.writerows(zip(*d2.values()))
    print('History saved.')

def set_loading_index(M, fname_dataset, reserve=0.02):
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/30sec/Spectrogram'
        m, _, _ = f[DataSpec].shape
    index = np.random.choice(np.arange(1,m), size=int(M * (2 + reserve)), replace=False)
    split = int(len(index)/2)
    index_test = index[split:]
    index_train_val = index[0:split]
    split_pct = 0.8
    split = int(split_pct * len(index_train_val))
    index_train = index_train_val[0:split]
    index_val = index_train_val[split:]
    return index_train, index_val, index_test
#
# def target_distribution(q):
#     """
#     Compute the target distribution p, given soft assignements, q. The target
#     distribtuion is generated by giving more weight to 'high confidence'
#     samples - those with a higher probability of being a signed to a certain
#     cluster. This is used in the KL-divergence loss function.
#     # Arguments
#         q: Soft assignement probabilities - Probabilities of each sample being
#            assigned to each cluster.
#     # Input:
#          2D tensor of shape [n_samples, n_features].
#     # Output:
#         2D tensor of shape [n_samples, n_features].
#     """
#     weight = q ** 2 / q.sum(0)
#     return (weight.T / weight.sum(1)).T
#
# def view_all_clusters(data, n, o, labels, n_clusters, sample_index, n_examples=6, show=True):
#     """
#     Shows six examples of spectrograms assigned to each cluster.
#     # Example
#     ```
#         print_all_clusters(data=X_train, labels=kmeans_labels, num_clusters=10)
#     ```
#     # Arguments
#         data: data set (4th rank tensor) that was used as the input for the clustering algorithm used.
#         labels: 1D array  of clustering assignments. The value of each label corresponds to the cluster
#                 that the samples in 'data' (with the same index) was assigned to. Array must be the same length as
#                 data.shape[0].
#         num_clusters: The number of clusters that the data was seperated in to.
#     # Input shape
#         2D tensor with shape: `(n_samples, n_features)`.
#     # Output shape
#         2D tensor with shape: `(n_samples, n_clusters)`.
#     """
#     n_rows = n_clusters
#     for k in range(0, n_clusters):
#         label_idx = np.where(labels==k)[0]
#         if len(label_idx) == 0:
#             n_rows -= 1
#
#     fig = plt.figure(figsize=(2*n_examples,2*n_rows), dpi=300)
#     gs = GridSpec(nrows=n_rows, ncols=n_examples)
#     cnt_row = 0
#     for i in range(0, n_clusters):
#         label_idx = np.where(labels==i)[0]
#         if len(label_idx) == 0:
#             pass
#         elif len(label_idx) < n_examples:
#             for j in range(0, len(label_idx)):
#                 ax = fig.add_subplot(gs[cnt_row,j])
#                 plt.imshow(np.reshape(data[label_idx[j],:,:,:], (n,o)), aspect='auto')
#                 plt.gca().invert_yaxis()
#                 if j == 0 and cnt_row == 0:
#                     plt.xlabel('Time Bins')
#                     plt.ylabel('Frequency Bins')
#                     plt.text(-0.1,1.3, f'Label {i}', weight='bold',transform=ax.transAxes)
#                 elif j == 0 and cnt_row != 0:
#                     plt.text(-0.1,1.3, f'Label {i}', weight='bold',transform=ax.transAxes)
#                 # print(j, type(j))
#                 # print(label_idx[j], type(label_idx[j])))
#                 plt.title(f'Index = {sample_index[label_idx[j]]}')
#             cnt_row += 1
#         else:
#             for j in range(0, n_examples):
#                 ax = fig.add_subplot(gs[cnt_row,j])
#                 plt.imshow(np.reshape(data[label_idx[j],:,:,:], (n,o)), aspect='auto')
#                 plt.gca().invert_yaxis()
#                 if j == 0 and cnt_row == 0:
#                     plt.xlabel('Time Bins')
#                     plt.ylabel('Frequency Bins')
#                     plt.text(-0.1,1.3, f'Label {i}', weight='bold',transform=ax.transAxes)
#                 elif j == 0 and cnt_row != 0:
#                     plt.text(-0.1,1.3, f'Label {i}', weight='bold',transform=ax.transAxes)
#                 plt.title(f'Index = {sample_index[label_idx[j]]}')
#             cnt_row += 1
#     fig.suptitle('Label Assignments', size=18,
#                  weight='bold')
#     # fig.set_size_inches(2*n_examples, 2*cnt_row)
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.92)
#     if show is False:
#         plt.close()
#     else:
#         plt.show()
#     return fig
#
def view_learningcurve(training_history, validation_history, show=True):
    epochs = len(training_history['mse'])
    fig = plt.figure(figsize=(18,6), dpi=300)
    gs = GridSpec(nrows=1, ncols=2)
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
#
# def view_LspaceRcnstr(X_val, val_enc, val_reconst, idx, n, o, fname_dataset,
#                       sample_index, figsize=(12,9), show=True):
#     fig = plt.figure(figsize=figsize, dpi=300)
#     gs = GridSpec(nrows=1, ncols=3, width_ratios=[1,.1,1])
#     # Original Spectrogram:
#     ax = fig.add_subplot(gs[0])
#     plt.imshow(np.reshape(X_val[idx,:, :, :], (n,o)), aspect='auto')
#     plt.ylabel('Frequency Bin')
#     plt.xlabel('Time Bin')
#     plt.gca().invert_yaxis()
#     plt.colorbar()
#     plt.title('Original Spectrogram')
#     # Latent Space Representation:
#     ax = fig.add_subplot(gs[1])
#     plt.imshow(val_enc[idx].reshape(32,1), cmap='viridis', aspect='auto')
#     plt.gca().invert_yaxis()
#     plt.title('Latent Space')
#     plt.tick_params(
#                     axis='x',          # changes apply to the x-axis
#                     which='both',      # both major and minor ticks are affected
#                     bottom=False,      # ticks along the bottom edge are off
#                     top=False,         # ticks along the top edge are off
#                     labelbottom=False) # labels along the bottom edge are off
#     # Reconstructed Spectrogram:
#     ax = fig.add_subplot(gs[2])
#     plt.imshow(np.reshape(val_reconst[idx,:, :, :], (n,o)), aspect='auto')
#     plt.ylabel('Frequency Bin')
#     plt.xlabel('Time Bin')
#     plt.gca().invert_yaxis()
#     plt.colorbar()
#     plt.title('Reconstructed Spectrogram')
#     fig.tight_layout()
#     if show is False:
#         plt.close()
#     else:
#         plt.show()
#     return fig
#

def view_specgram_training(fixed_images, reconstructed_images, n, o, figtitle,
                           figsize=(12,9), show=True):
    X_T = fixed_images.detach().cpu().numpy()
    X_V = reconstructed_images.detach().cpu().numpy()
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(nrows=2, ncols=4)
    counter = 0
    for i in range(fixed_images.size()[0]):
        ax = fig.add_subplot(gs[0,counter])
        plt.imshow(np.reshape(X_T[i,:,:,:], (n,o)), aspect='auto')
        plt.gca().invert_yaxis()
        plt.xlabel('Time Bin')
        plt.ylabel('Frequency Bin')
        if counter == 0:
            plt.figtext(-0.01, 0.62, 'Original Spectrograms', rotation='vertical',
                        fontweight='bold')

        ax = fig.add_subplot(gs[1,counter])
        plt.imshow(np.reshape(X_V[i,:,:,:], (n,o)), aspect='auto')
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

# def view_orig_rcnstr_specgram(X_val, val_reconst, insp_idx, n, o,
#                               fname_dataset, sample_index, figtitle, nrows=2,
#                               ncols=4, figsize=(12,9), show=True):
#     '''Plots selected spectrograms and their latent space reconstructions.'''
#     if not len(insp_idx) == (nrows * ncols / 2):
#         raise ValueError('Subplot/sample number mismatch: check dimensions.')
#     metadata = get_metadata(insp_idx, sample_index, fname_dataset)
#     fig = plt.figure(figsize=figsize, dpi=300)
#     gs = GridSpec(nrows=nrows, ncols=ncols)
#     counter = 0
#     for i in range(len(insp_idx)):
#         ax = fig.add_subplot(gs[0,counter])
#         plt.imshow(np.reshape(X_val[insp_idx[i],:,:,:], (n,o)), aspect='auto')
#         plt.gca().invert_yaxis()
#         plt.ylabel('Frequency Bins')
#         plt.xlabel('Time Bins')
#         station = metadata[counter]['Station']
#         try:
#             time_on = datetime.strptime(metadata[counter]['TriggerOnTime'],
#                                         '%Y-%m-%dT%H:%M:%S.%f').strftime(
#                                         '%Y-%m-%dT%H:%M:%S.%f')[:-4]
#         except:
#             time_on = datetime.strptime(metadata[counter]['TriggerOnTime'],
#                                         '%Y-%m-%dT%H:%M:%S').strftime(
#                                         '%Y-%m-%dT%H:%M:%S.%f')[:-4]
#         plt.title(f'Station {station}\nTrigger: {time_on}\n'
#                   f'Index: {sample_index[insp_idx[i]]}')
#         if counter == 0:
#             plt.figtext(0, 0.57, 'Original Spectrograms', rotation='vertical',
#                         fontweight='bold')
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         plt.colorbar(cax=cax)
#
#         ax = fig.add_subplot(gs[1,counter])
#         plt.imshow(np.reshape(val_reconst[insp_idx[i],:,:,:], (n,o)),
#                               aspect='auto')
#         plt.gca().invert_yaxis()
#         plt.ylabel('Frequency Bins')
#         plt.xlabel('Time Bins')
#         if counter == 0:
#             plt.figtext(0, 0.15, 'Reconstructed Spectrograms',
#                         rotation='vertical', fontweight='bold')
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         plt.colorbar(cax=cax)
#         counter += 1
#
#     fig.suptitle('Spectrograms Reconstructed from Latent Space', size=18,
#                  weight='bold')
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.85, left=0.05)
#     if show is False:
#         plt.close()
#     else:
#         plt.show()
#     return fig

def view_specgram(X, insp_idx, n, o, fname_dataset, sample_index, figtitle,
                  nrows=2, ncols=2, figsize=(12,9), show=True):
    '''Plots selected spectrograms from input data.'''
    if not len(insp_idx) == nrows * ncols:
        raise ValueError('Subplot/sample number mismatch: check dimensions.')
    metadata = get_metadata(insp_idx, sample_index, fname_dataset)
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(nrows=nrows, ncols=ncols)
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
    if show is False:
        plt.close()
    else:
        plt.show()
    return fig
