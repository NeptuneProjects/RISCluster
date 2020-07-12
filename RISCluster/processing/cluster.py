
from contextlib import redirect_stdout
from datetime import datetime
import json
import random
import sys
sys.path.insert(0, '../../RISCluster/')

import h5py
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as tfkb

from RISCluster.utils.notify import notify

class ClusteringLayer(tf.keras.layers.Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a
    vector that represents the probability of the sample belonging to each
    cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)`
        which represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        #initialize object attributes
        self.n_clusters = n_clusters
        self.alpha = alpha #exponent for soft assignment calculation
        self.initial_weights = weights
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tfkb.floatx(),
                                                    shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters',
                                        shape=(self.n_clusters,
                                        int(input_dim)),
                                        initializer='glorot_uniform')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (tfkb.sum(tfkb.square(tfkb.expand_dims(inputs,
                                   axis=1) - self.clusters), axis=2) \
                                   / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tfkb.transpose(tfkb.transpose(q) / tfkb.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConvAEC:
    def build(m, n, o, p, depth, strides, activation, kernel_init, latent_dim):

        # Define Input Dimensions:
        img_input = tf.keras.layers.Input(shape=(n,o,p))

        # Define the encoder:
        e = tf.keras.layers.Conv2D(depth*2**0, (5,5), strides = strides,
                        activation=activation, kernel_initializer=kernel_init,
                        padding='same')(img_input)
        e = tf.keras.layers.Conv2D(depth*2**1, (5,5), strides = strides,
                        activation=activation, kernel_initializer=kernel_init,
                        padding='same')(e)
        e = tf.keras.layers.Conv2D(depth*2**2, (3,3), strides = strides,
                        activation=activation, kernel_initializer=kernel_init,
                        padding='same')(e)
        e = tf.keras.layers.Conv2D(depth*2**3, (3,3), strides = strides,
                        activation=activation, kernel_initializer=kernel_init,
                        padding='same')(e)
        shape_before_flattening = tf.keras.backend.int_shape(e)
        e = tf.keras.layers.Flatten()(e)
        encoded = tf.keras.layers.Dense(latent_dim, activation=activation,
                                        name='encoded')(e)

        # Define the decoder:
        d = tf.keras.layers.Dense(np.prod(shape_before_flattening[1:]),
                                  activation='relu')(encoded)
        d = tf.keras.layers.Reshape(shape_before_flattening[1:])(d)
        d = tf.keras.layers.Conv2DTranspose(depth*2**2, (3,3), strides=strides,
                        activation=activation, kernel_initializer=kernel_init,
                        padding='same')(d)
        d = tf.keras.layers.Conv2DTranspose(depth*2**1, (5,5), strides=strides,
                        activation=activation, kernel_initializer=kernel_init,
                        padding='same')(d)
        d = tf.keras.layers.Conv2DTranspose(depth*2**0, (5,5), strides=strides,
                        activation=activation, kernel_initializer=kernel_init,
                        padding='same')(d)
        decoded = tf.keras.layers.Conv2DTranspose(1, (5,5), strides=strides,
                        activation='linear', kernel_initializer=kernel_init,
                        padding='same')(d)

        # Create the models (Encoder, Autoencoder):
        encoder = tf.keras.models.Model(inputs=img_input, outputs=encoded,
                                        name='Encoder')
        autoencoder = tf.keras.models.Model(inputs=img_input, outputs=decoded,
                                            name='Autoencoder')
        return encoder, autoencoder

def build_dtvec(starttime, dt):
    pass
    return None

def build_fvec(cutoff, fs):
    pass
    return None

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

def get_trace():
    pass
    return None

def load_test(fname_dataset, M, index_test):
    with h5py.File(fname_dataset, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/30sec/Spectrogram'
        dset = f[DataSpec]
        m, n, o = dset.shape
        print(f'H5 file has {m} samples, {n} frequency bins, {o} time bins.')
        print(f'Loading {M} test samples...')
        tic = datetime.now()

        np.seterr(divide='raise')
        X = np.empty([M, n-2, o-173, 1])
        sample_index = np.empty([M,], dtype=np.int)
        dset_arr = np.empty([1, n, o])
        count = 0
        for i in range(M):
            try:
                dset_arr = dset[index_test[i], 1:-1, 1:129]
                dset_arr /= dset_arr.max()
                X[count,:,:,:] = dset_arr[..., np.newaxis]
                sample_index[count,] = int(index_test[i])
                count += 1
            except:
                print('Numpy "Divide-by-zero Warning" raised, '
                      'skipping spectrogram.')
                pass

            print('%.2f' % (float(100*i/(M-1))) + '% complete.', end='\r')
        toc = datetime.now()
        print(f'\nTime elapsed = {toc}')

    # Update dimensions of X:
    m, n, o, p = X.shape
    print(f'Shape of X is {(m, n, o, p)}')
    msgsubj = 'Training/Validation Data Loaded'
    msgcontent = f'''{M} training/validation spectrograms loaded successfully.
Time Elapsed = {(toc-tic)}'''
    notify(msgsubj, msgcontent)
    return X, m, n, o, p, sample_index

def load_train_val(fname_dataset, M):
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
        print(f'H5 file has {m} samples, {n} frequency bins, {o} time bins.')
        print(f'Loading {M} training/validation samples...')
        tic = datetime.now()

        index_train_val, index_test = set_load_index(m, M, reserve=0.02)

        np.seterr(divide='raise')
        X = np.empty([M, n-2, o-173, 1])
        sample_index = np.empty([M,], dtype=np.int)
        dset_arr = np.empty([1, n, o])
        count = 0
        for i in range(M):
            try:
                dset_arr = dset[index_train_val[i], 1:-1, 1:129]
                dset_arr /= dset_arr.max()
                X[count,:,:,:] = dset_arr[..., np.newaxis]
                sample_index[count,] = int(index_train_val[i])
                count += 1
            except:
                print('Numpy "Divide-by-zero Warning" raised, '
                      'skipping spectrogram.')
                pass

            print('%.2f' % (float(100*i/(M-1))) + '% complete.', end='\r')
        toc = datetime.now()
        print(f'\nTime elapsed = {toc}')

    # Update dimensions of X:
    m, n, o, p = X.shape
    print(f'Shape of X is {(m, n, o, p)}')
    msgsubj = 'Training/Validation Data Loaded'
    msgcontent = f'''{M} training/validation spectrograms loaded successfully.
Time Elapsed = {(toc-tic)}'''
    notify(msgsubj, msgcontent)
    return X, m, n, o, p, sample_index, index_train_val, index_test

# Traceback (most recent call last):
#   File "DEC.py", line 276, in <module>
#     update_interval, labels, labels_last)
#   File "../../RISCluster/RISCluster/processing/cluster.py", line 288, in optim_and_cluster
#     loss = model.train_on_batch(x=X_train[idx], y=[p[idx], X_train[idx,:,:,:]])
#   File "/Users/williamjenkins/miniconda3/envs/RIS_Cluster/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py", line 1175, in train_on_batch
#     outputs = self.train_function(ins)  # pylint: disable=not-callable
#   File "/Users/williamjenkins/miniconda3/envs/RIS_Cluster/lib/python3.7/site-packages/tensorflow/python/keras/backend.py", line 3292, in __call__
#     run_metadata=self.run_metadata)
#   File "/Users/williamjenkins/miniconda3/envs/RIS_Cluster/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1458, in __call__
#     run_metadata_ptr)
# tensorflow.python.framework.errors_impl.InvalidArgumentError: input and filter must have the same depth: 1 vs 8

def optim_and_cluster(X_train, model, batch_size, tol, maxiter,
                      update_interval, labels, labels_last):
    loss = 0              # initialize loss
    index = 0             # initialize index to start
    tic = datetime.now()
    loss_list = np.zeros([maxiter,3])                       # Keep track of loss function during training process
    print('============================================')
    print(f'Size of X_train = {X_train.shape}')
    index_array = np.arange(X_train.shape[0])
    print(f'Size of index_array = {index_array.shape}')
    for ite in range(int(maxiter)):
        print('---------------------------------------')
        print(f'ite = {ite}')
        if ite % update_interval == 0:
            q, reconst  = model.predict(X_train, verbose=1) # Calculate soft assignment distribtuion & CAE reconstructions
            p = target_distribution(q)                      # Update the auxiliary target distribution p
            labels = q.argmax(1)                            # Assign labels to the embedded latent space samples
            # check stop criterion - Calculate the % of labels that changed from previous update
            delta_label = np.sum(labels != labels_last).astype(np.float32) /labels.shape[0]
            labels_last = np.copy(labels)                   # Generate copy of labels for future updates
            loss= np.round(loss, 5)                         # Round the loss

            print('Iter %d' % ite)
            print('Loss: {}'.format(loss))
            print_cluster_size(labels)                      # Show the number of samples assigned to each cluster

            if ite > 0 and delta_label < tol:               # Break training if loss reaches the tolerance threshhold
                print('delta_label ', delta_label, '< tol ', tol)
                break

        idx = index_array[index * batch_size: min((index+1) * batch_size, X_train.shape[0])]

        print(f'idx = {idx}')
        print(f'Size of X_train[idx] = {X_train[idx].shape}')
        print(f'Size of p[idx] = {p[idx].shape}')
        loss = model.train_on_batch(x=X_train[idx], y=[p[idx], X_train[idx,:,:,:]])
        # print(f'Size of loss = {loss.shape}')
        index = index + 1 if (index + 1) * batch_size < X_train.shape[0] else 0
        print(f'(Updated) index = {index}')
        print('---------------------------------------')
    toc = datetime.now()
    print(f'Deep Embedded Clustering Computation Time: {toc-tic}')
    msgsubj = 'DEC Training Complete'
    msgcontent = f'''DEC training completed at {datetime.now()}.
Time Elapsed = {toc-tic}.'''
    notify(msgsubj, msgcontent)
    return model, reconst

def print_cluster_size(labels):
    """
    Shows the number of samples assigned to each cluster.
    # Example
    ```
        print_cluster_size(labels=kmeans_labels)
    ```
    # Arguments
        labels: 1D array  of clustering assignments. The value of each label corresponds to the cluster
                that the samples in the clustered data set (with the same index) was assigned to. Array must be the same length as
                data.shape[0]. where 'data' is the clustered data set.
    """
    num_labels = max(labels) + 1
    for j in range(0,num_labels):
        label_idx = tf.where(labels==j)[0]
        print("Label " + str(j) + ": " + str(label_idx.shape[0]))

def save_DEC_labelInds(fname, n_clusters, labels):
    with h5py.File(fname, 'w') as nf:
        for i in range(0, n_clusters):
            fname = 'label_{0}_indices'.format(i)
            label_idx = np.where(labels==i)[0]
            nf.create_dataset(fname, data=label_idx, dtype=label_idx.dtype)

def save_DEC_lspace(fname, enc_test):
    with h5py.File(fname, 'w') as nf:
        nf.create_dataset('EncodedData', data=enc_test, dtype=enc_test.dtype)

def save_model_info(model, fname):
    with open(fname + '.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    tf.keras.utils.plot_model(model, to_file=fname + '.png', show_shapes=True)

def save_trained_lspace(fname, train_enc, val_enc, val_reconst):
    with h5py.File(fname, 'w') as nf:
        nf.create_dataset('Train_EncodedData', data=train_enc,
                          dtype=train_enc.dtype)
        nf.create_dataset('Val_EncodedData', data=val_enc,
                          dtype=val_enc.dtype)
        nf.create_dataset('Val_Reconst', data=val_reconst,
                          dtype=val_reconst.dtype)

def set_load_index(m, M, reserve=0.02):
    index = np.random.choice(m, size=int(M * (2 + reserve)), replace=False)
    index_split = int(len(index)/2)
    index_train_val = sorted(index[0:index_split])
    index_test = sorted(index[index_split:])
    return index_train_val, index_test

def target_distribution(q):
    """
    Compute the target distribution p, given soft assignements, q. The target
    distribtuion is generated by giving more weight to 'high confidence'
    samples - those with a higher probability of being a signed to a certain
    cluster. This is used in the KL-divergence loss function.
    # Arguments
        q: Soft assignement probabilities - Probabilities of each sample being
           assigned to each cluster.
    # Input:
         2D tensor of shape [n_samples, n_features].
    # Output:
        2D tensor of shape [n_samples, n_features].
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def view_all_clusters(data, n, o, labels, n_clusters, sample_index, n_examples=6, show=True):
    """
    Shows six examples of spectrograms assigned to each cluster.
    # Example
    ```
        print_all_clusters(data=X_train, labels=kmeans_labels, num_clusters=10)
    ```
    # Arguments
        data: data set (4th rank tensor) that was used as the input for the clustering algorithm used.
        labels: 1D array  of clustering assignments. The value of each label corresponds to the cluster
                that the samples in 'data' (with the same index) was assigned to. Array must be the same length as
                data.shape[0].
        num_clusters: The number of clusters that the data was seperated in to.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
    n_rows = n_clusters
    for k in range(0, n_clusters):
        label_idx = np.where(labels==k)[0]
        if len(label_idx) is 0:
            n_rows -= 1

    fig = plt.figure(figsize=(2*n_examples,2*n_rows), dpi=300)
    gs = GridSpec(nrows=n_rows, ncols=n_examples)
    cnt_row = 0
    for i in range(0, n_clusters):
        label_idx = np.where(labels==i)[0]
        if len(label_idx) is 0:
            pass
        elif len(label_idx) < n_examples:
            for j in range(0, len(label_idx)):
                ax = fig.add_subplot(gs[cnt_row,j])
                plt.imshow(np.reshape(data[label_idx[j],:,:,:], (n,o)), aspect='auto')
                plt.gca().invert_yaxis()
                if j == 0 and cnt_row == 0:
                    plt.xlabel('Time Bins')
                    plt.ylabel('Frequency Bins')
                    plt.text(-0.1,1.3, f'Label {i}', weight='bold',transform=ax.transAxes)
                elif j == 0 and cnt_row != 0:
                    plt.text(-0.1,1.3, f'Label {i}', weight='bold',transform=ax.transAxes)
                # print(j, type(j))
                # print(label_idx[j], type(label_idx[j])))
                plt.title(f'Index = {sample_index[label_idx[j]]}')
            cnt_row += 1
        else:
            for j in range(0, n_examples):
                ax = fig.add_subplot(gs[cnt_row,j])
                plt.imshow(np.reshape(data[label_idx[j],:,:,:], (n,o)), aspect='auto')
                plt.gca().invert_yaxis()
                if j == 0 and cnt_row == 0:
                    plt.xlabel('Time Bins')
                    plt.ylabel('Frequency Bins')
                    plt.text(-0.1,1.3, f'Label {i}', weight='bold',transform=ax.transAxes)
                elif j == 0 and cnt_row != 0:
                    plt.text(-0.1,1.3, f'Label {i}', weight='bold',transform=ax.transAxes)
                plt.title(f'Index = {sample_index[label_idx[j]]}')
            cnt_row += 1
    fig.suptitle('Label Assignments', size=18,
                 weight='bold')
    # fig.set_size_inches(2*n_examples, 2*cnt_row)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    if show is False:
        plt.close()
    else:
        plt.show()
    return fig

def view_learningcurve(fname_logger, show=True):
    hist = np.genfromtxt(fname_logger, delimiter=',', skip_header=1,
                         names=['epoch', 'train_mse_loss', 'train_mae_loss',
                         'val_mse_loss', 'val_mae_loss'])
    fig = plt.figure(figsize=(20,6), dpi=300)
    gs = GridSpec(nrows=1, ncols=2)
    ax = fig.add_subplot(gs[0])
    plt.plot(hist['epoch'], hist['train_mae_loss'], label='Training Loss')
    plt.plot(hist['epoch'], hist['val_mae_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.title('Mean Absolute Error')
    plt.legend()

    ax = fig.add_subplot(gs[1])
    plt.plot(hist['epoch'], hist['train_mse_loss'], label='Training Loss')
    plt.plot(hist['epoch'], hist['val_mse_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.title('Mean Squared Error')
    plt.legend()
    fig.tight_layout()
    if show is False:
        plt.close()
    else:
        plt.show()
    return fig

def view_LspaceRcnstr(X_val, val_enc, val_reconst, idx, n, o, fname_dataset,
                      sample_index, figsize=(12,9), show=True):
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(nrows=1, ncols=3, width_ratios=[1,.1,1])
    # Original Spectrogram:
    ax = fig.add_subplot(gs[0])
    plt.imshow(np.reshape(X_val[idx,:, :, :], (n,o)), aspect='auto')
    plt.ylabel('Frequency Bin')
    plt.xlabel('Time Bin')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Original Spectrogram')
    # Latent Space Representation:
    ax = fig.add_subplot(gs[1])
    plt.imshow(val_enc[idx].reshape(32,1), cmap='viridis', aspect='auto')
    plt.gca().invert_yaxis()
    plt.title('Latent Space')
    plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
    # Reconstructed Spectrogram:
    ax = fig.add_subplot(gs[2])
    plt.imshow(np.reshape(val_reconst[idx,:, :, :], (n,o)), aspect='auto')
    plt.ylabel('Frequency Bin')
    plt.xlabel('Time Bin')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Reconstructed Spectrogram')
    fig.tight_layout()
    if show is False:
        plt.close()
    else:
        plt.show()
    return fig

def view_orig_rcnstr_specgram(X_val, val_reconst, insp_idx, n, o,
                              fname_dataset, sample_index, figtitle, nrows=2,
                              ncols=4, figsize=(12,9), show=True):
    '''Plots selected spectrograms and their latent space reconstructions.'''
    if not len(insp_idx) == (nrows * ncols / 2):
        raise ValueError('Subplot/sample number mismatch: check dimensions.')
    metadata = get_metadata(insp_idx, sample_index, fname_dataset)
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(nrows=nrows, ncols=ncols)
    counter = 0
    for i in range(len(insp_idx)):
        ax = fig.add_subplot(gs[0,counter])
        plt.imshow(np.reshape(X_val[insp_idx[i],:,:,:], (n,o)), aspect='auto')
        plt.gca().invert_yaxis()
        plt.ylabel('Frequency Bins')
        plt.xlabel('Time Bins')
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
        if counter == 0:
            plt.figtext(0, 0.57, 'Original Spectrograms', rotation='vertical',
                        fontweight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = fig.add_subplot(gs[1,counter])
        plt.imshow(np.reshape(val_reconst[insp_idx[i],:,:,:], (n,o)),
                              aspect='auto')
        plt.gca().invert_yaxis()
        plt.ylabel('Frequency Bins')
        plt.xlabel('Time Bins')
        if counter == 0:
            plt.figtext(0, 0.15, 'Reconstructed Spectrograms',
                        rotation='vertical', fontweight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        counter += 1

    fig.suptitle('Spectrograms Reconstructed from Latent Space', size=18,
                 weight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.85, left=0.05)
    if show is False:
        plt.close()
    else:
        plt.show()
    return fig

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
        plt.imshow(np.reshape(X[insp_idx[i],:,:,:], (n,o)), aspect='auto')
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
