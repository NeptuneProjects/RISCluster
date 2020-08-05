# Deep Convolutional Embedded Clustering
This repository is a PyTorch implementation of deep convolutional embedded clustering (DCEC) in the unsupervised classification of seismic signals detected on a 34-station passive broadband seismic array on the Ross Ice Shelf from November 2014 to November 2016.  The workflow requires the following elements:
1. Import modules and set up environment
2. Load and pre-process data
3. Construct a convolutional auto-encoder (AEC)
4. Tune, train, and validate AEC
5. Determine optimal number of clusters using gap statistic analysis
6. Incorporate Clustering Layer into AEC model architecture
7. Simultaneously train the DCEC model to maximize clustering while fine tuning AEC parameters.
8. Feed unseen data to DCEC model and evaluate performance.

## References:
Master thesis of Dylan Snover: https://github.com/dsnover/Unsupervised_Machine_Learning_for_Urban_Seismic_Noise
<br>Xie, Girshick, and Farhadi (2016): https://arxiv.org/abs/1511.06335)
<br>Guo, Zhu, Liu, and Jin (2017): https://link.springer.com/chapter/10.1007/978-3-319-70096-0_39

Project assembled by William Jenkins
<br>wjenkins [@] ucsd [dot] edu
<br>http://noiselab.ucsd.edu
<br>Scripps Institution of Oceanography
<br>University of California San Diego
<br>La Jolla, California, USA
