#!/usr/bin/env python3

import argparse

from RISCluster import utils

def main(args):
    M = args.M
    fname_dataset = args.path
    savepath = args.savepath
    utils.save_TraVal_index(M, fname_dataset, savepath)


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

    main(args)
