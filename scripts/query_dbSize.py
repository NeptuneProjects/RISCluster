import argparse

import h5py

def query_dbSize(path):
    with h5py.File(path, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/30sec/Spectrogram'
        dset = f[DataSpec]
        m, n, o = dset.shape
        m -= 1

    print(f' >> h5 dataset contains {m} samples with dimensions [{n},{o}]. <<')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter path to .h5 file.')
    parser.add_argument(
        'path',
        help='Enter path to database; must be .h5/.hd5 file.'
    )
    args = parser.parse_args()
    path = args.path
    query_dbSize(path)
