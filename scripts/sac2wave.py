from datetime import datetime, timedelta
import importlib as imp
import os
import sys
sys.path.insert(0, '../RISCluster/')

import obspy
from obspy import read, read_inventory, Stream, UTCDateTime
import pandas as pd

import processing
imp.reload(processing)

datadir='/Users/williamjenkins/Research/Data/RIS_Seismic/HDH'
respf = [f for f in os.listdir(datadir) if 'RESP' in f][0]
files = sorted([f for f in os.listdir(datadir) if 'HDH' in f])
files = files[0:10]


dt_start = processing.file2dt(files[0]).date()
dt_end = processing.file2dt(files[-1]).date()
dti = pd.date_range(dt_start, dt_end, freq='D')

NFFT = 4096
overlap = 0.25
taper_trace = 10
pre_feed = 20
cutoff = 0.4
buffer_front = taper_trace + pre_feed
buffer_back = taper_trace

# To-do: Parallelize this loop:
for d in range(1, len(dti) - 1):
    t0 = dti[d]
    t1 = dti[d+1]
    time_start = t0 - timedelta(minutes=buffer_front)
    time_stop = t1 + timedelta(minutes=buffer_back)
    search_start = time_start.floor('D')
    search_stop = time_stop.ceil('D')
    search_range = pd.date_range(search_start, search_stop,freq='D')

    first = True
    for i in range(len(search_range) - 1):
        flist = [f for f in files if processing.file2dt(f).date() in search_range[0:-1]]
        for f, fname in enumerate(flist):
            if first:
                st = read(f"{datadir}/{fname}")
                first = False
            else:
                st += read(f"{datadir}/{fname}")
    st.merge(method=1, fill_value='interpolate', interpolation_samples=5)
    tr = st[0].trim(starttime=UTCDateTime(time_start), endtime=UTCDateTime(time_stop))

    tr.detrend(type='linear')
    tr.taper(max_percentage=0.5, type='hann', max_length=60*taper_trace)
    fs = tr.stats.sampling_rate
    # Decimate:
    tr.filter("lowpass", freq=0.4, corners=2, zerophase=True)
    tr.decimate(100, no_filter=True)
    # Remove Instrument Response:
    inv = obspy.read_inventory(f"{datadir}/{respf}", format='RESP')
    pre_filt = (0.00005, 0.00006, 0.5, 0.6)
    tr.remove_response(inventory=inv, water_level=15, output='DISP',
                       pre_filt=pre_filt, plot=True)


    # seedresp = {'filename': respf,'units': 'DIS'}
    # tr.simulate(paz_remove=None, seedresp=seedresp)
