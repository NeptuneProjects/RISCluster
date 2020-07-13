from datetime import datetime
import os
import random
import sys
sys.path.insert(0, '../../RISCluster/')

import h5py
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import tensorflow as tf

from RISCluster.processing import cluster
from RISCluster.utils.utils import notify

import importlib
importlib.reload(cluster)

tic_total = datetime.now()
# AEC Parameters:
M = int(2400)
LR = 0.0001     # Learning rate
n_epochs = 600  # Number of epochs
batch_sz = 256  # Batch size

# ==== 1. Set Up Environment ==================================================
fname_dataset = '../../../Data/DetectionData.h5'
savepath_fig = '../../../Outputs/Figures/'
savepath_stats = '../../../Outputs/Metrics/'
savepath_model = '../../../Outputs/Models/'
savepath_data = '../../../Outputs/SavedData/'
todays_date = datetime.now().strftime('%Y%m%d')
seed = 2009
# random.seed(seed)
# np.random.seed(seed)
cluster.init_GPU(GPU_frac=1.0)

# ==== 2. Load and Pre-process Data ===========================================
# Define number of samples to be read into memory:
# M = int(5e5)
# M = int(2400)

# Load and reformat data:
X, m, n, o, p, sample_index, _, index_test = \
                                       cluster.load_train_val(fname_dataset, M)
# Select random sample indices for inspection:
insp_idx = sorted(np.random.randint(0,len(X),4))
# sample_index = [65, 1940, 4418, 4476]
# insp_idx = [1, 2, 3, 4]
# Output 1: View randomly selected spectrograms:
figtitle = 'Input Spectrograms'
fig = cluster.view_specgram(X, insp_idx, n, o, fname_dataset, sample_index,
                            figtitle, nrows=2, ncols=2, figsize=(12,9),
                            show=False,)
fname = savepath_fig + '01_InputSpecGrams_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.png'
fig.savefig(fname)

# Split data and clean up memory:
X_train, X_val = train_test_split(X, test_size=0.2, shuffle=True,
                                  random_state=seed)
insp_idx = sorted(np.random.randint(0,len(X_val),4))
del X

# ==== 3. Construct Convolutional Autoencoder =================================
# Convolutional autoencoders (CAE) are designed to compress images to a lower
# dimensional latent space, and then reconstruct them based on these latent
# features.  During the training process, the encoder learns to extract salient
# features of the original image so that reconstruction results in the lowest
# loss.  Once the model is trained, the latent features representing the
# original images can be used to cluster the samples into separable classes.

# Define common parameters to the encoder/decoder:
depth = 8
strides = 2
activation = 'relu'
kernel_init = 'glorot_uniform'
latent_dim = 32

# Construct the autoencoder model.  Save it and the encoder model.
# Input: Spectrograms
# Output: Compressed image representations (embedded latent space)
encoder, autoencoder = cluster.ConvAEC.build(m, n, o, p, depth, strides,
                                         activation, kernel_init, latent_dim)
autoencoder.summary()
fname = savepath_model + '01_ConvAEC_Architecture_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S")
cluster.save_model_info(autoencoder, fname)

# ==== 4. Tune, Train, & Validate Autoencoder =================================
# To ensure the model is learning important features and not overfitting
# (memorizing the data) or underfitting(generalizing the data), we need to set
# hyperparameters that affect the model's training. The learning rate, batch
# size, number of epochs are the primary factors that effect learning. The
# parameters below have been fine tuned through a long process of trial and
# error. An additional step we take to prevent overfitting is the use of
# validation data that is not used to train the model, but is used to validate
# the models performance after each epoch. The loss for the validation data is
# used as an indicator of over fitting. If the loss for the validation data
# increases for conecutive epochs, the model is overfitting. We implement an
# early stopping criteria that halts training and restores the best model
# parameters if the validation data does not decrease for 10 consectutive
# epochs.

# LR = 0.0001     # Learning rate
# n_epochs = 600  # Number of epochs
# batch_sz = 128  # Batch size
# Adaptive learning rate optimization algorithm (Adam)
optim = tf.keras.optimizers.Adam(lr=LR)
# Mean squared error loss function
loss = 'mse'
print(f'LR = {LR}\nbatch_sz = {batch_sz}')

# Create log file to record training & validation loss:
fname_logger = savepath_stats + '01_LearningCurve_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.csv'
csv_logger = tf.keras.callbacks.CSVLogger(fname_logger)

# Early stopping halts training after validation loss stops decreasing for 10
# consectutive epochs:
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                           patience=10, verbose=1, mode='min',
                                           restore_best_weights=True)

# Compile the Model:
# Compile encoder & autoencoder(initialize random filter weights):
encoder.compile(loss=loss,optimizer=optim)
autoencoder.compile(loss=loss, optimizer=optim,
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Run Training and Display Training Loss (Learning Curve) for
# Hyperparameter Validation:
tic=datetime.now()
autoencoder.fit(X_train, X_train, batch_size=batch_sz, epochs=n_epochs,
                validation_data=(X_val, X_val),
                callbacks=[csv_logger, early_stop])
toc = datetime.now()
print(f'Elapsed Time: {toc - tic}')

msgsubj = 'ConvAEC Training Complete'
msgcontent = f'''ConvAEC training completed at {toc}.
Time Elapsed = {(toc-tic)}.'''
notify(msgsubj, msgcontent)

# Save initial model architectures with weights:
fname = savepath_model + '01_ConvAEC_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.hdf5'
autoencoder.save(fname)
fname = savepath_model + '02_ConvEncoder_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.hdf5'
encoder.save(fname)

# Plot the training and validation losses:
# Retrieve training/validation metrics from file:
fig = cluster.view_learningcurve(fname_logger, show=False)
fname = savepath_fig + '02_MSE_MAE_LossLrnCrv_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.png'
fig.savefig(fname)

# Show Validation Data Reconstructions:
# We can observe how well the model is learning important features in the input
# spectrograms by looking at the autoencoder's ability to reconstruct the
# original input images from the validation data set.  Because we are
# compressing each spectrogram image from 8192 features down to 32, we expect
# to see a fair amount of smoothing of features in the reconstructed images. A
# well trained autoencoder will retain most of the amplitude information
# (observed in the color bar scales) and will preserve local structure (the
# strongest signals are seen in both the original and reconstructed images).

# Reconstruction of validation data:
val_reconst = autoencoder.predict(X_val, verbose = 1)
# Embedded latent space samples of validation data:
val_enc = encoder.predict(X_val, verbose = 1)
# Embedded latent space samples of training data:
train_enc = encoder.predict(X_train, verbose = 1)
# Save latent space data:
fname = savepath_data + '01_TrngValLSpace_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.hdf5'
cluster.save_trained_lspace(fname, train_enc, val_enc, val_reconst)
figtitle = 'Validation Data Reconstruction from Latent Space'
fig = cluster.view_orig_rcnstr_specgram(X_val, val_reconst, insp_idx, n, o,
                                        fname_dataset, sample_index, figtitle,
                                        nrows=2, ncols=4, figsize=(14,9),
                                        show=False)
fname = savepath_fig + '03_RcnstrSpecgram-Val_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.png'
fig.savefig(fname)

# Show original spectrogram, embedded representation, and reconstruction:
idx = insp_idx[0]
fig = cluster.view_LspaceRcnstr(X_val, val_enc, val_reconst, idx, n, o,
                          fname_dataset, sample_index, figsize=(12,9),
                          show=False)
fname = savepath_fig + '04_LspaceRcnst_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.png'
fig.savefig(fname)

# ==== 5. Determine Optimal Number of Clusters Using Gap Statistic Analysis ===
# Perform gap statistic calculation. The gap statistic is a measure of
# clustering performance that compares how well a data set can be clustered
# against a reference distribution. Reference distributions are those where no
# meaning meaningful clusters exist(i.e. a normal or uniform distrubtion - both
# were tested here).
# Additional information about the gap statistic can be found in the following
# paper: https://statweb.stanford.edu/~gwalther/gap
#
# Create Reference Distributions:
# For sake of time and computational efficiency-take 50,000 random samples from
# training data to compute gap statistic:
#
# M_gap = 100
# rand_idx = np.random.randint(0,len(X_train),M_gap)
# kmeans_enc = np.zeros([M_gap,32])
# for i in range(len(rand_idx)):
#     kmeans_enc[i] = train_enc[rand_idx[i]]
# print(kmeans_enc.shape)
#
# ==== 6. Incorporate Clustering Layer into Model Architecture ================
# Deep embedded clustering is a method that seeks to improve clustering
# performance based on the cluster assignment of each embedded latent space
# sample. The method is considered a fine tuning step as it assumes the initial
# centroid locations found from KMeans is accurate. The model parameters are
# updated in a way that maximizes seperability  between embedded latent samples
# assigned to different groups. For more information: the following paper
# describes the algoirth in detail:http://proceedings.mlr.press/v48/xieb16.pdf
# NOTE: The clustering layer architecture and training algorithm were derived
# from code written as part of the above paper. The full code can be found at:
# https://github.com/XifengGuo/DEC-keras/blob/master/DEC.py
# Minor changes were made to the code to tailor the algorithm to the needs of
# this research.
#
# Define the clustering layer in the overall model architecture:
n_clusters = 11
clustering_layer = cluster.ClusteringLayer(n_clusters,
                                           name='clustering')(encoder.output)
ClusterModel = tf.keras.models.Model(inputs=autoencoder.input,
                              outputs=[clustering_layer, autoencoder.output])
ClusterModel.compile(loss=['kld',loss], loss_weights=[0.1, .9],
                     optimizer=optim)
# ClusterModel.summary()
# fname = savepath_model + '03_ConvAECwClustering_' + \
        # datetime.now().strftime("%Y%m%dT%H%M%S")
# cluster.save_model_info(ClusterModel, fname)

### 6.2 Generate embedded latent space training samples:
enc_train = encoder.predict(X_train, verbose=1)

# Run KMeans with n_clusters, run 100 initializations to ensure accuracy:
kmeans = KMeans(n_clusters=n_clusters, n_init=100)

# Get initial assignments and make copy of labels for future reference (see
# DEC training below):
labels = kmeans.fit_predict(enc_train)
labels_last = np.copy(labels)

# Initialize the DEC clustering layer weights using cluster centers found
# initally by KMeans:
ClusterModel.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# ==== 7. Train the DEC Model =================================================
# Set parameters for the DEC fine tuning:
batch_size = batch_sz      # number of samples in each batch
tol = 0.001           # tolerance threshold to stop training
maxiter = 47250       # number of updates to run before halting. (~12 epochs)
update_interval = 315 # Soft assignment distribution and target distributions
                      # updated evey 315 batches. (~12 updates/epoch)
ClusterModel, reconst = cluster.optim_and_cluster(X_train, ClusterModel, batch_size, tol,
                                           maxiter, update_interval, labels,
                                           labels_last)
#Save model and model weights seperately
fname = savepath_model + '04_DEC_ModelWeightsFinal_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S")
ClusterModel.save_weights(fname)
fname = savepath_model + '05_DEC_Model_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S")
ClusterModel.save(savepath_model + 'DEC_model_{}.hdf5'.format(todays_date))

insp_idx = sorted(np.random.randint(0,len(X_train),4))
figtitle = 'Training Data Reconstruction from DEC Latent Space'
fig = cluster.view_orig_rcnstr_specgram(X_train, reconst, insp_idx, n, o,
                                fname_dataset, sample_index, figtitle, nrows=2,
                                ncols=4, figsize=(12,9), show=False)
fname = savepath_fig + '05_RcnstrSpecgram-DEC-Trn_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.png'
fig.savefig(fname)

# ==== 8. Evaluate Performance of DEC Model on Unseen Test Data ===============
# Use trained DEC model to predict clustering assignments and generate
# reconstuctions of  input spectrogram images from the unseen test data set. The
# test data set was isolated from the training data set at the beginning of the
# workflow.
#
# Load in test data:
X_test, m, n, o, p, sample_index = cluster.load_test(fname_dataset, M,
                                                     index_test)
# Feed Unseen Test Data to Trained DEC Model
# Predict asignment probability of test data & generate reconstructions:
q, reconst = ClusterModel.predict(X_test, verbose = 1)
# Determine labels based on assignment probabilities:
labels = q.argmax(1)
# Generate embedded latent space test data samples:
enc_test = encoder.predict(X_test)
# Examples of Original Spectrograms and their Reconstructions:
insp_idx = sorted(np.random.randint(0,len(X_test),4))
figtitle = 'Test Data Reconstruction from DEC Latent Space'
fig = cluster.view_orig_rcnstr_specgram(X_test, reconst, insp_idx, n, o,
                                fname_dataset, sample_index, figtitle, nrows=2,
                                ncols=4, figsize=(12,9), show=False)
fname = savepath_fig + '06_RcnstrSpecgram-DEC-Tst_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.png'
fig.savefig(fname)

# Examples of Clustering Assignments:
# To estimate how well the clustering algorithm is grouping similar
# spectrograms, we look at 5 examples of spectrograms that were assigned to
# each cluster. A well performing DEC model will group spectrorgams that are
# visually similar.  We also show the size of each cluster to ensure that each
# cluster is being utilized in the algorithm  and that one cluster is not
# dominating all other assignments. If either of these conditions are apperent,
# it is an indication that the DEC model is converging to a trivial solution.
#
# Display the number of samples assigned to each cluster:
cluster.print_cluster_size(labels)
# Show six examples of sample spectrogram assigned to each cluster:
fig = cluster.view_all_clusters(X_test, n, o, labels, n_clusters, sample_index,
                                n_examples=6, show=False)
fname = savepath_fig + '07_LabeledSpecgrams_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.png'
fig.savefig(fname)
# Save Clustering Data
# Save the index of all samples assigned to each cluster in seperate data sets
# for future analysis.
fname = savepath_data + '02_DEC_LabelInds_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.hdf5'
cluster.save_DEC_labelInds(fname, n_clusters, labels)
# Save embedded latent space test data for future analysis:
fname = savepath_data + '03_DEC_LSpaceTest_' + \
        datetime.now().strftime("%Y%m%dT%H%M%S") + '.hdf5'
cluster.save_DEC_lspace(fname, enc_test)

# ==== End of Script ==========================================================
toc_total = datetime.now()
print(f'Workflow complete at {toc_total}.  '
      'Time elapsed: {toc_total-tic_total}.')
msgsubj = 'DEC Script Complete'
msgcontent = f'''DEC script completed successfully at {toc_total}.
Time Elapsed = {toc_total-tic_total}'''
notify(msgsubj, msgcontent)
