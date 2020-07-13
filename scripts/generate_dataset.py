"""Generate Dataset for Machine Learning Applications

William Jenkins, wjenkins@ucsd.edu
Scripps Institution of Oceanography, UC San Diego
July 2020
This script generates samples of traces and spectrograms for use in later
machine learning applications.  The samples are detected using recursive
STA/LTA, then saved to .h5 as datasets with metadata.  This script is
optimized for parallel processing.
"""

import concurrent.futures
from datetime import datetime
import json
import sys
sys.path.insert(0, '../../RISCluster/')

import h5py
import numpy as np

from RISCluster.processing import processing as process
from RISCluster.utils.utils import notify

# ========================== Initialize Parameters ============================
# v v v v Modify these parameters when switching to Velella! v v v v
num_workers = 17
# datadir = '/Volumes/RISData/' # <----- Edit directory containing data.
datadir = '/home/wfjenkin/Research/Data/RIS_Seismic/'
# ^ ^ ^ ^ Modify these parameters when switching to Velella! ^ ^ ^ ^
network_index = 0
station_index = np.arange(0, 34)
channel_index = 2
T_seg = 30 # Duration of traces & spectrograms (sec)
NFFT = 128
overlap = 0.9 # Fraction of overlap for spectrograms
taper_trace = 10 # Taper (min)
pre_feed = 20 # Buffer (min)
cutoff = 3.0 # Highpass cutoff frequency (Hz)
STA = 1 # Short-term Average Window (sec)
LTA = 8 # Long-term Average Window (sec)
trigger_on = 6.0 # STA/LTA trigger-on threshold
trigger_off = 5.0 # STA/LTA trigger-off threshold
data_savepath = '../../../Data/' # Location of .h5 file
data_savename = 'DetectionData_Small.h5' # File name
group_name = str(T_seg) + 'sec' # Group Name (grouped by T_seg)
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

# ====================== Wrapper Function for Workflow ========================
def workflow_wrapper(station_index):
    signal_args = process.SigParam(datadir, network_index, station_index,
                                   channel_index, datetime_index, taper_trace,
                                   pre_feed, cutoff, T_seg, NFFT, overlap)
    detector_args = process.DetectorParam(STA, LTA, trigger_on, trigger_off)
    try:
        tr, signal_args = process.readStream_addBuffer(signal_args)
        inv = process.read_stationXML(signal_args)
        tr, signal_args = process.trace_processor(tr, inv, signal_args)
        tr_out, S_out, catdict = process.detector_recstalta(tr, signal_args,
                                                            detector_args)
        return tr_out, S_out, catdict
    except:
        print(f'    Station {process.get_station(station_index)} skipped.')
        return None, None, None

# ================================ Run Script =================================
if __name__ == '__main__':
    tic_run = datetime.now()

    for i in range(numdays):
        tic = datetime.now()
        datetime_index = datetime_indexes[i]

        if i is not 0:
            print('    ----------------------------------------------------')
        print(f'    Processing {i + 1}/{len(datetime_indexes)}: '
              f'{process.get_datetime(datetime_index)} (Day {datetime_index})')

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
            as executor:
            results = list(executor.map(workflow_wrapper, station_index))
            j = 0

            for r in results:

                if any(map(lambda x: x is None, r)) is True:
                    pass
                elif j == 0:
                    tr = r[0]
                    S = r[1]
                    dict = r[2]
                    j += 1
                else:
                    tr = np.append(tr, r[0], axis=0)
                    S = np.append(S, r[1], axis=0)
                    dict.extend(r[2])
                    j += 1

        with h5py.File(data_savepath+data_savename, 'a') as f:

            if ('/' + group_name) not in f:
                print(f'    No h5 group found, creating group "{group_name}" '
                      'and datasets.')
                h5group_name = f.create_group(group_name)
                h5group_name.attrs['T_seg (s)'] = T_seg
                h5group_name.attrs['NFFT'] = NFFT
                dset_tr, dset_spec, dset_cat = \
                                    process.get_datasets(T_seg, NFFT, 100,
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
                dset_cat[-m+j,] = json.dumps(dict[j])

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
