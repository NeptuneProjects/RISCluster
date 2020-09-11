import argparse

import h5py

from utils import query_dbSize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter path to .h5 file.')
    parser.add_argument(
        'path',
        help='Enter path to database; must be .h5/.hd5 file.'
    )
    args = parser.parse_args()
    path = args.path
    m, n, o = query_dbSize(path)
    print(f" >> h5 dataset contains {m} samples with dimensions [{n},{o}]. <<")
