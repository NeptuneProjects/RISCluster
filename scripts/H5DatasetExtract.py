import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import json
import os
import sys
sys.path.insert(0, '../RISCluster/')

import argparse
import h5py
import numpy as np
from tqdm import tqdm

import importlib as imp
import processing
imp.reload(processing)
import utils

debug = False
if debug:
    source = '../../../Data/Full/Full.h5'
    include = '[24]'
    exclude = None
    after = None
    before = None
    dest = '../../../Data/RS09.h5'

if __name__ == "__main__":
    if not debug:
        parser = argparse.ArgumentParser(
            description="Creates new dataset from existing dataset."
        )
        parser.add_argument("source", help="Enter path to source dataset.")
        parser.add_argument("dest", help="Enter path to destination dataset.")
        parser.add_argument("--include", help="Enter stations to include.")
        parser.add_argument("--exclude", help="Enter stations to exclude.")
        parser.add_argument("--after", help="Include after YYYYMMDDTHHMMSS.")
        parser.add_argument("--before", help="Include before YYYYMMDDTHHMMSS.")
        args = parser.parse_args()

        if args.include is None and args.exclude is None:
            raise Exception("Must specify stations to include or exclude.")
        source = args.source
        dest = args.dest
        include = args.include
        exclude = args.exclude
        after = args.after
        before = args.before

    if not os.path.exists(source):
        raise ValueError(f"Source file not found: {source}")

    with h5py.File(source, 'r') as rf:
        M = len(rf['/4s/Trace'])
    index = np.arange(1, M)

    if include is not None:
        include = json.loads(include)
    else:
        include = None
    if exclude is not None:
        exclude = json.loads(exclude)
    else:
        exclude = None
    if after is not None:
        after = after
    else:
        after = None
    if before is not None:
        before = before
    else:
        before = None

    if include is not None and exclude is not None:
        removals = [processing.get_station(i) for i in exclude]
        stations = [processing.get_station(i) for i in include]
        stations = [i for i in stations if i not in removals]
        print(f"Searching {stations}")
    elif include is not None:
        stations = [processing.get_station(i) for i in include]
        print(f"Searching {stations}")
    elif exclude is not None:
        removals = [processing.get_station(i) for i in exclude]
        stations = [processing.get_station(i) for i in range(34)]
        stations = [i for i in stations if i not in removals]
        print(f"Searching {stations}")
    else:
        stations = [processing.get_station(i) for i in range(34)]
        print(f"Searching {stations}")

    A = [{
            "index": index[i],
            "source": source,
            "stations": stations
        } for i in range(len(index))]

    index_keep = np.zeros(len(index))
    with ProcessPoolExecutor(max_workers=14) as exec:
        print("Finding detections that meet filter criteria...")
        futures = [exec.submit(processing._find_indeces, **a) for a in A]
        kwargs1 = {
            "total": int(len(index)),
            "bar_format": '{l_bar}{bar:20}{r_bar}{bar:-20b}',
            "leave": True
        }
        for i, future in enumerate(tqdm(as_completed(futures), **kwargs1)):
            index_keep[i] = future.result()
    index_keep = np.sort(index_keep[~np.isnan(index_keep)]).astype(int)

    with h5py.File(source, 'r') as fs, h5py.File(dest, 'w') as fd:
        M = len(index_keep)
        dset_names = ['Catalogue', 'Trace', 'Spectrogram', 'Scalogram']
        # dset_names = ['Catalogue']
        for dset_name in dset_names:
            group_path = '/4s'
            dset = fs[f"{group_path}/{dset_name}"]
            dset_shape = dset.shape[1:]
            dset_shape = (M,) + dset_shape
            group_id = fd.require_group(group_path)
            dset_id = group_id.create_dataset(dset_name, dset_shape, dtype=dset.dtype, chunks=None)
            processing._copy_attributes(dset, dset_id)
            kwargs2 = {
                "desc": dset_name,
                "bar_format": '{l_bar}{bar:20}{r_bar}{bar:-20b}'
            }
            for i in tqdm(range(len(index_keep)), **kwargs2):
                dset_id[i] = dset[index_keep[i]]
