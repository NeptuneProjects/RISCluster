# RISCluster
RISCluster is a package that implements **deep embedded clustering** (DEC) of
seismic data recorded on the Ross Ice Shelf, Antarctica from 2014-2017. This
package is an accompaniment to a paper submitted to the Journal of Geophysical Research (Jenkins II et al., submitted).

RISCluster is in the process of being restructured so that it can be easily installed
and run on a Mac or Linux environment.

This repository is a PyTorch implementation of DEC. The workflow is as follows:
1. Load and pre-process data
2. Construct a convolutional auto-encoder (AEC)
3. Tune, train, and validate AEC
4. Incorporate clustering layer into AEC model architecture
5. Intialize clusters (K-Means, GMM, K-Medioids available)
6. Train the DEC model: clustering and model training are simultaneous.
7. Once trained, infer class labels for remainder of the data set.

### Installation
Pre-requisites:
[Anaconda](https://anaconda.org) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Tested on MacOS 11.1 and Red Hat Enterprise Linux 7.9.

The following steps will set up a Conda environment and install RISProcess.
1. Open a terminal and navigate to the directory you would like to download the
 **RISCluster.yml** environment file.
2. Save **RISCluster.yml** to your computer by running the following:
  <br>a. **Mac**:
  <br>`curl -LJO https://raw.githubusercontent.com/NeptuneProjects/RISCluster/master/RISClusterMacOS.yml`
  <br>b. **Linux**:
  <br>`wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/NeptuneProjects/RISCluster/master/RISCluster.yml`
3. In terminal, run: `conda env create -f RISCluster.yml`
4. Once the environment is set up and the package is installed, activate your
environment by running `conda activate RISCluster` in terminal.

### References
*Submitted*: William F. Jenkins II, Peter Gerstoft, Michael J. Bianco, Peter D. Bromirski; *Unsupervised Deep Clustering of Seismic Data: Monitoring the Ross Ice Shelf, Antarctica.* Submitted to Journal of Geophysical Research on 20 Jan 2021; doi: https://doi.org/10.1002/essoar.10505894.1

Dylan Snover, Christopher W. Johnson, Michael J. Bianco, Peter Gerstoft; *Deep Clustering to Identify Sources of Urban Seismic Noise in Long Beach, California.* Seismological Research Letters 2020; doi: https://doi.org/10.1785/0220200164

Junyuan Xie, Ross Girshick, Ali Farhadi; *Unsupervised Deep Embedding for Clustering Analysis.* Proceedings of the 33rd International Conference on Machine Learning, New York, NY, 2016; https://arxiv.org/abs/1511.06335v2

### Author
Project assembled by William Jenkins
<br>wjenkins [@] ucsd [dot] edu
<br>Scripps Institution of Oceanography
<br>University of California San Diego
<br>La Jolla, California, USA
