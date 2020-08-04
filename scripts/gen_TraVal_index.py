import argparse
import os
import sys
sys.path.insert(0, '../RISCluster/')

import importlib as imp
import utils
imp.reload(utils)

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Enter sample number.')
    my_parser.add_argument(
        'M',
        metavar='M',
        type=int,
        help='Enter number of spectrograms to be used for training/validation.'
    )
    args = my_parser.parse_args()
    M = args.M
    fname_dataset = '../../../Data/DetectionData.h5'
    savepath = '../../../Data/'
    utils.save_TraVal_index(M, fname_dataset, savepath)
