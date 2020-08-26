"""Generate Dataset for Machine Learning Applications

William Jenkins, wjenkins@ucsd.edu
Scripps Institution of Oceanography, UC San Diego
July 2020
This script generates samples of traces and spectrograms for use in later
machine learning applications.  The samples are detected using recursive
STA/LTA, then saved to .h5 as datasets with metadata.  This script is
optimized for parallel processing.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import sys
sys.path.insert(0, '../RISCluster/')

import h5py
import numpy as np
from tqdm import tqdm

import processing as process
from processing import workflow_wrapper
from utils import notify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Enter number of CPUs to be used."
    )
    parser.add_argument('num_workers', help="Enter number of workers.")
    args = parser.parse_args()
    # ========================== Initialize Parameters ============================
    # v v v v Modify these parameters when switching to Velella! v v v v
    # num_workers = 12
    num_workers = args.num_workers
    datadir = '/Volumes/RISData/' # <----- Edit directory containing data.
    # datadir = '/home/wfjenkin/Research/Data/RIS_Seismic/'
    # ^ ^ ^ ^ Modify these parameters when switching to Velella! ^ ^ ^ ^
    network_index = 0
    station_index = np.arange(0, 34)
    channel_index = 2
    T_seg = 4 # Duration of traces & spectrograms (sec)
    NFFT = 128
    tpersnap = 1 / 4
    overlap = 0.92 # Fraction of overlap for spectrograms
    taper_trace = 10 # Taper (min)
    pre_feed = 20 # Buffer (min)
    cutoff = 3.0 # Highpass cutoff frequency (Hz)
    STA = 1 # Short-term Average Window (sec)
    LTA = 8 # Long-term Average Window (sec)
    trigger_on = 6.0 # STA/LTA trigger-on threshold
    trigger_off = 5.0 # STA/LTA trigger-off threshold
    data_savepath = '../../../Data/' # Location of .h5 file
    group_name = str(T_seg) + 's' # Group Name (grouped by T_seg)
    data_savename = f'DetectionData_{group_name}.h5' # File name
    print('======================================================================')
    print('Select range of experiment days to compute. Stop day is midnight of the'
          '\nnew day and does NOT include that day\'s data.')
    day_start = int(input('Start Day: '))
    print(f' - Day {day_start} selected for start.')
    day_stop = int(input('Stop Day:  '))
    print(f' - Day {day_stop} selected for stop.')
    print('----------------------------------------------------------------------')
    datetime_indexes = np.arange(day_start, day_stop)
    numdays = len(datetime_indexes)
    # ================================ Run Script =================================
    tic_run = datetime.now()
    for i in range(numdays):
        tic = datetime.now()
        datetime_index = datetime_indexes[i]

        if i != 0:
            print('    ----------------------------------------------------')
        print(f'    Processing {i + 1}/{len(datetime_indexes)}: '
              f'{process.get_datetime(datetime_index)} (Day {datetime_index})')

        A = [dict(
            station_index=station_index[k],
            datadir=datadir,
            network_index=network_index,
            channel_index=channel_index,
            datetime_index=datetime_index,
            taper_trace=taper_trace,
            pre_feed=pre_feed,
            cutoff=cutoff,
            T_seg=T_seg,
            NFFT=NFFT,
            tpersnap=tpersnap,
            overlap=overlap,
            STA=STA,
            LTA=LTA,
            trigger_on=trigger_on,
            trigger_off=trigger_off,
            debug=False
        ) for k in range(len(station_index))]

        with ProcessPoolExecutor(max_workers=num_workers) as exec:
            futures = [exec.submit(workflow_wrapper, **a) for a in A]
            # results = list(exec.map(lambda p: workflow_wrapper(*p), params))
            # results = list(executor.map(workflow_wrapper, station_index))
            kwargs = {
                'total': int(len(futures)),
                'unit': 'station',
                'unit_scale': True,
                'bar_format': '{l_bar}{bar:20}{r_bar}{bar:-20b}',
                'leave': True
            }

            for future in tqdm(as_completed(futures), **kwargs):
                future.result()

        print('    Collecting results...')
        j = 0
        for future in futures:
            output = future.result()

            if any(map(lambda x: x is None, output)) is True:
                pass
            elif j == 0:
                tr = output[0]
                S = output[1]
                metadata = output[2]
            else:
                tr = np.append(tr, output[0], axis=0)
                S = np.append(S, output[1], axis=0)
                metadata.extend(output[2])
            j += 1

        print('    Saving results...')
        with h5py.File(data_savepath+data_savename, 'a') as f:
            if ('/' + group_name) not in f:
                print(f'    No h5 group found, creating group "{group_name}" '
                      'and datasets.')
                h5group_name = f.create_group(group_name)
                h5group_name.attrs['T_seg (s)'] = T_seg
                h5group_name.attrs['NFFT'] = NFFT
                dset_tr, dset_spec, dset_cat = \
                                    process.get_datasets(T_seg, NFFT, tpersnap, 100,
                                                         h5group_name, overlap)

            m = tr.shape[0]
            print(f'    {m} detections found.')
            dset_tr = f['/' + group_name + '/Trace']
            dset_tr.resize(dset_tr.shape[0]+m, axis=0)
            dset_tr[-m:,:] = tr
            dset_spec = f['/' + group_name + '/Spectrogram']
            dset_spec.resize(dset_spec.shape[0]+m, axis=0)
            dset_spec[-m:,:,:] = S
            dset_cat = f['/' + group_name + '/Catalogue']
            dset_cat.resize(dset_cat.shape[0]+m, axis=0)

            for j in np.arange(0,m):
                dset_cat[-m+j,] = json.dumps(metadata[j])

    toc_run = datetime.now() - tic_run
    print('----------------------------------------------------------------------')
    print(f'Processing complete at {datetime.now()}; {toc_run} elapsed for '
          f'{numdays} day(s) processed.')
    print(f'Number of parallel workers: {num_workers}')
    print(f'Start Day = {day_start}, Stop Day = {day_stop}')
    print('*Note: Stop day is midnight of the new day and does NOT include that \n'
          'day\'s data. To resume computation, use Stop Day as the new Start Day.')
    subj = 'Seismic Data Pre-processing Job Complete'
    msg = f'''Pre-processing completed at {datetime.now()}.
    Number of parallel workers = {num_workers}
    Start Day = {day_start}, Stop Day = {day_stop}
    Time Elapsed = {toc_run}\n
    *Note: Stop day is midnight of the new day and does NOT include
    that day\'s data.  To resume computation, use Stop Day as the
    new Start Day.'''
    notify(subj,msg)
    print('======================================================================')

# For debugging:
# import importlib as imp
# imp.reload(process)
# tr_out, S_out, catdict = [workflow_wrapper(**a) for a in A]
