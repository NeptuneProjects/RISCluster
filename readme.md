# Deep Embedded Clustering
This project aims to implement deep embedded clustering (DEC) in the unsupervised classification of seismic signals detected on a 34-station passive broadband seismic array on the Ross Ice Shelf from November 2014 to November 2016.  The workflow contains the following elements:
1. Import modules and set up environment
2. Load and pre-process data
3. Construct the convolutional auto-encoder (CAE)
4. Tune, train, and validate CAE
5. Determine optimal number of clusters using gap statistic analysis
6. Incorporate Clustering Layer into Model Architecture
7. Train the DEC Model
8. Evaluate Performance of DEC Model

This work has been heavily adapted from Dylan Snover's master's thesis workflow.

Project assembled by William Jenkins
Scripps Institution of Oceanography
University of California San Diego
La Jolla, California, USA

###  Project Outline:
## 1.  Load Data
Load data using custom loader from spectrograms; place these data into dataloader instances for use by PyTorch.

## 2.  Pre-train Convolutional Autoencoder
# 2.1  Instantiate the autoencoder:
encoder = cluster.Encoder
decoder = cluster.Decoder
autoencoder = cluster.AEC(encoder, decoder)

# 2.2  Train the autoencoder:
600 epochs or until losses stabilize.

# 2.3  Save AEC weights:
Save the model's learned parameters to file for future use.  In particular, the encoder's weights will be used to initialize the DNN portion of the DEC model.

# 3. Train Deep Embedded Clustering Model
