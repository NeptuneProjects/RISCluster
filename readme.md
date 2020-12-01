# Deep Embedded Clustering
This repository is a PyTorch implementation of deep embedded clustering (DEC) for unsupervised classification of seismic signals.  The workflow was specifically tailored to data detected on a 34-station passive broadband seismic array on the Ross Ice Shelf, Antarctica from November 2014 to November 2016.  The workflow requires the following elements:
1. Import modules and set up environment
2. Load and pre-process data
3. Construct a convolutional auto-encoder (AEC)
4. Tune, train, and validate AEC
5. Incorporate clustering layer into AEC model architecture
6. Intialize clusters (K-Means, GMM, K-Medioids available)
7. Train the DEC model: clustering and model training are simultaneous.
8. Once trained, infer class labels for remainder of the data set.

## References:
Master thesis of Dylan Snover: https://github.com/dsnover/
<br>Xie, Girshick, and Farhadi (2016): https://arxiv.org/abs/1511.06335)

Project assembled by William Jenkins
<br>wjenkins [@] ucsd [dot] edu
<br>http://noiselab.ucsd.edu
<br>Scripps Institution of Oceanography
<br>University of California San Diego
<br>La Jolla, California, USA
