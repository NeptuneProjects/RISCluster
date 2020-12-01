import argparse
import os
import sys
sys.path.insert(0, '../RISCluster/')

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter sample number.')
    parser.add_argument(
        'M',
        metavar='M',
        type=int,
        help='Enter number of spectrograms to be used for training/validation.'
    )
    parser.add_argument(
        'path',
        metavar='path',
        help='Enter path to h5 dataset.'
    )
    parser.add_argument(
        'savepath',
        metavar='savepath',
        help='Enter savepath'
    )
    args = parser.parse_args()
    M = args.M
    fname_dataset = args.path
    savepath = args.savepath
    utils.save_TraVal_index(M, fname_dataset, savepath)
