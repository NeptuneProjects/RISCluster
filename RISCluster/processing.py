from datetime import datetime, timedelta
import json
import logging
import os
import random
from subprocess import run

import h5py
import numpy as np
import numpy.ma as ma
from obspy import read, read_inventory, Stream, UTCDateTime
from obspy.clients.fdsn.mass_downloader import MassDownloader, \
    RectangularDomain, Restrictions
from obspy.signal.tf_misfit import cwt
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, \
    trigger_onset
import pandas as pd
from scipy import signal
from tqdm import tqdm


class DetectorParam():
    def __init__(self, STA, LTA, trigger_on, trigger_off):
        self.STA = STA
        self.LTA = LTA
        self.trigger_on = trigger_on
        self.trigger_off = trigger_off


class SigParam():
    def __init__(
            self,
            datadir,
            network_index,
            station_index,
            channel_index,
            datetime_index,
            taper_trace,
            pre_feed,
            cutoff,
            T_seg,
            NFFT,
            tpersnap,
            overlap
        ):
        self.datadir = datadir
        self.network_index = network_index
        self.station_index = station_index
        self.channel_index = channel_index
        self.datetime_index = datetime_index
        self.taper_trace = taper_trace
        self.pre_feed = pre_feed
        self.cutoff = cutoff
        self.T_seg = T_seg
        self.NFFT = NFFT
        self.tpersnap = tpersnap
        self.overlap = overlap


def _copy_attributes(in_object, out_object):
    '''Copy attributes between 2 HDF5 objects.'''
    for key, value in in_object.attrs.items():
        out_object.attrs[key] = value


def _find_indeces(index, source, stations):
    with h5py.File(source, 'r') as f:
        metadata = json.loads(f['/4s/Catalogue'][index])
    if metadata["Station"] in stations:
        return index
    else:
        return np.nan


def detector(tr, signal_args, detector_args, detector_type='classic'):
    '''Detect events using recursive STA/LTA.'''
    fs = signal_args.fs
    STA = int(detector_args.STA*fs)
    LTA = int(detector_args.LTA*fs)
    trigger_on = detector_args.trigger_on
    trigger_off = detector_args.trigger_off
    buffer_front = signal_args.buffer_front
    buffer_back = signal_args.buffer_back
    T_seg = signal_args.T_seg
    NFFT = signal_args.NFFT
    tpersnap = signal_args.tpersnap
    overlap = signal_args.overlap
    dtvec = signal_args.dtvec
    network = signal_args.network
    station = signal_args.station
    channel = signal_args.channel
    cutoff = signal_args.cutoff

    if detector_type == 'classic':
        cft = classic_sta_lta(tr.data, STA, LTA)
    elif detector_type == 'recursive':
        cft = recursive_sta_lta(tr.data, STA, LTA)

    on_off = trigger_onset(cft, trigger_on, trigger_off)
    # If no detections found, function returns None.
    if type(on_off) is list:
        raise Exception('Unable to process station.')
        # return None, None, None
    t0_index = int(buffer_front*60*fs)
    t1_index = int(tr.stats.npts - buffer_back*60*fs - 1)
    on_off = on_off[((on_off >= t0_index) & (on_off < t1_index))[:, 0], :]
    on = on_off[:, 0]
    off = on_off[:, 1]
    # If no detections found, function returns None.
    if on.size == 0:
        # print('No detections')
        raise Exception('Unable to process station.')
        # return None, None, None
    # Analyze and save data from each detection:
    tr_out = np.zeros((len(on), int(fs*T_seg + 1)))
    S_out = np.zeros((len(on), 70, int(T_seg//(tpersnap*(1 - overlap)) + 1)))
    # S_out = np.empty((len(on), int(NFFT/2 + 1) + 1,
    #                  int(T_seg//(tpersnap*(1 - overlap)) + 1)))
    C_out = np.zeros((len(on), 69, int(T_seg//(tpersnap*(1 - overlap)))))
    # C_out = np.empty((len(on), int(NFFT/2 + 1),
                     # int(T_seg//(tpersnap*(1 - overlap)))))
    catdict = [{
        "Network": None,
        "Station": None,
        "Channel": None,
        "StartTime": None,
        "EndTime": None,
        "SamplingRate": None,
        "SamplingInterval": None,
        "Npts": None,
        "TriggerOnThreshold": None,
        "TriggerOffThreshold": None,
        "TriggerOnTime": None,
        "TriggerOffTime": None
    } for i in range(len(on))]

    for i in range(len(on)):
        # Index is 1 +/- 1/8 sec before trigger:
        i0 = int(on[i] - int(fs*1) + random.randint(int(-fs*1/8), int(fs*1/8)))
        # i0 = int(on[i])
        # Index2 is T_seg sec after i0:
        i1 = int(i0 + T_seg*fs)
        # Trim and normalize trace:
        tr_sample = tr.copy().trim(starttime=UTCDateTime(dtvec[i0]),
                       endtime=UTCDateTime(dtvec[i1]))
        # Normalize trace to [-1, 1]:
        tr_data = tr_sample.data
        # tr_data /= np.abs(tr_data).max()
        tr_out[i, :] = tr_data
        # Calculate spectrogram:
        t_spec, f_spec, S = get_specgram(
            tr,
            fs,
            i0,
            i1,
            NFFT,
            overlap,
            tpersnap
        )
        t_spec = np.insert(t_spec, 0, np.NAN)
        S = np.hstack((f_spec[:, None], S))
        S_out[i, :, :] = np.vstack((S, t_spec))
        # Calculate CWT:
        C_out[i, :, :] = get_cwt(tr_data, fs, cutoff)
        # Save metadata into dictionary:
        catdict[i] = {
           "Network": network,
           "Station": station,
           "Channel": channel,
           "StartTime": tr_sample.stats.starttime.isoformat(),
           "EndTime": tr_sample.stats.endtime.isoformat(),
           "SamplingRate": tr_sample.stats.sampling_rate,
           "SamplingInterval": tr_sample.stats.delta,
           "Npts": tr_sample.stats.npts,
           "TriggerOnThreshold": trigger_on,
           "TriggerOffThreshold": trigger_off,
           "TriggerOnTime": dtvec[on[i]].isoformat(),
           "TriggerOffTime": dtvec[off[i]].isoformat()
        }

    return tr_out, S_out, C_out, catdict


def file2dt(fname):
    fname = fname.split('.')[0:5]
    dt = datetime.strptime(
        f'{fname[0]} {fname[1]} {fname[2]} {fname[3]} {fname[4]}',
        '%Y %j %H %M %S'
    )
    return dt


def get_channel(channel_index):
    '''Input: Integer channel index (0-2).
       Output: Channel name (str)'''
    channel_list = ['HHE', 'HHN', 'HHZ']
    channel_name = channel_list[channel_index]
    return channel_name


def get_cwt(tr, fs, cutoff):
    scalogram = cwt(tr, 1 / fs, 8, cutoff, 30, 69)
    return np.abs(scalogram[:,0:-1:2])


def get_datasets(T_seg, NFFT, tpersnap, fs, group_name, overlap):
    '''Defines the structure of the .h5 database; h5py package required.
    Of note, pay special attention to the dimensions of the chunked data. By
    anticipating the output data dimensions, one can chunk the saved data on
    disk accordingly, making the reading process go much more quickly.'''
    # Set up dataset for traces:
    m = 1
    n = fs*T_seg + 1
    dset_tr = group_name.create_dataset(
        'Trace',
        (m, n),
        maxshape=(None,n),
        chunks=(20, n),
        dtype='f'
    )
    dset_tr.attrs['AmplUnits'] = 'Velocity (m/s)'
    # Set up dataset for spectrograms:
    m = 1
    # n = int(NFFT/2 + 1) + 1
    n = 70
    o = int(T_seg/(tpersnap*(1 - overlap)) + 1)
    dset_spec = group_name.create_dataset(
        'Spectrogram',
        (m, n, o),
        maxshape=(None, n, o),
        chunks=(20, n, o),
        dtype='f'
    )
    dset_spec.attrs['TimeUnits'] = 's'
    dset_spec.attrs['TimeVecXCoord'] = np.array([1,200])
    dset_spec.attrs['TimeVecYCoord'] = 70
    dset_spec.attrs['FreqUnits'] = 'Hz'
    dset_spec.attrs['FreqVecXCoord'] = 0
    dset_spec.attrs['FreqVecYCoord'] = np.array([0,68])
    dset_spec.attrs['AmplUnits'] = '(m/s)^2/Hz'
    # Set up dataset for scalograms:
    m = 1
    # n = int(NFFT/2 + 1)
    n = 69
    o = int(T_seg/(tpersnap*(1 - overlap)))
    dset_scal = group_name.create_dataset(
        'Scalogram',
        (m, n, o),
        maxshape=(None, n, o),
        chunks=(20, n, o),
        dtype='f'
    )

    # Set up dataset for catalogue:
    m = 1
    dtvl = h5py.string_dtype(encoding='utf-8')
    dset_cat = group_name.create_dataset(
        'Catalogue',
        (m,),
        maxshape=(None,),
        dtype=dtvl
    )
    return dset_tr, dset_spec, dset_scal, dset_cat


def get_datetime(datetime_index):
    '''Input: Integer datetime index for any day between ti and tf.
       Output: Datetime string'''
    ti = "20141202T000000"
    tf = "20161129T000000"
    datetimes = pd.date_range(ti, tf, freq='d')
    datetime = datetimes[datetime_index]
    return datetime


def get_metadata(query_index, sample_index, fname_dataset):
    '''Returns station metadata given sample index.'''
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/4s/Catalogue'
        dset = f[DataSpec]
        metadata = dict()
        counter = 0
        for i in query_index:
            query = sample_index[i]
            metadata[counter] = json.loads(dset[query])
            counter += 1
    return metadata


def get_network(network_index):
    '''Input: Integer network index (0).
       Output: Network name string'''
    network_list = ['XH']
    network_name = network_list[network_index]
    return network_name


def get_specgram(tr, fs, i0, i1, NFFT, overlap, tpersnap):
    npersnap = fs*tpersnap # Points per snapshot.
    # if npersnap % 2: # If the index is odd, change to an even index.
    #     npersnap = npersnap + 1
    i0 = int(i0 - npersnap/2)
    i1 = int(i1 + npersnap/2 - 1)
    tr = tr[i0:i1]
    tr /= np.abs(tr).max()
    window = np.kaiser(npersnap, beta=5.7)
    f, t, S = signal.spectrogram(
        tr,
        fs,
        window,
        nperseg=npersnap,
        noverlap = overlap*npersnap,
        nfft = NFFT
    )
    t = t - min(t)
    mask = (f >= 3) & (f <= 30)
    S = S[mask, : ]
    f = f[mask]
    return t, f, S


def get_station(station_index):
    '''Input: Integer station index (0-33).
       Output: Station name string'''
    station_list = ['DR01', 'DR02', 'DR03', 'DR04', 'DR05', 'DR06', 'DR07',
                    'DR08', 'DR09', 'DR10', 'DR11', 'DR12', 'DR13', 'DR14',
                    'DR15', 'DR16', 'RS01', 'RS02', 'RS03', 'RS04', 'RS05',
                    'RS06', 'RS07', 'RS08', 'RS09', 'RS10', 'RS11', 'RS12',
                    'RS13', 'RS14', 'RS15', 'RS16', 'RS17', 'RS18']
    station_name = station_list[station_index]
    return station_name


def mass_data_downloader(
        savepath,
        start='20141201',
        stop='20161201',
        Network='XH',
        Station='*',
        Channel='HH*'
    ):
    '''
    This function uses the FDSN mass data downloader to automatically download
    data from the XH network deployed on the RIS from Nov 2014 - Nov 2016.
    More information on the Obspy mass downloader available at:
    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html
    Inputs:
    savepath: "[string to savepath]"
    start: "YYYYMMDD"
    stop:  "YYYYMMDD"
    Network: 2-character FDSN network code
    Station: 2-character station code
    Channel: 3-character channel code
    *Note: Location is set to "*" by default.

    William Jenkins
    Scripps Institution of Oceanography, UC San Diego
    February 2020
    '''
    print("=" * 65)
    print("Initiating mass download request.")
    start = UTCDateTime(start)
    stop  = UTCDateTime(stop)

    if not os.path.exists(savepath):
        os.makedirs(f'{savepath}/MSEED')
        os.makedirs(f'{savepath}/StationXML')

    domain = RectangularDomain(
        minlatitude=-85,
        maxlatitude=-75,
        minlongitude=160,
        maxlongitude=-130
    )

    restrictions = Restrictions(
        starttime = start,
        endtime = stop,
        chunklength_in_sec = 86400,
        network = Network,
        station = Station,
        location = "*",
        channel = Channel,
        reject_channels_with_gaps = True,
        minimum_length = 0.0,
        minimum_interstation_distance_in_m = 100.0
    )

    mdl = MassDownloader(providers=["IRIS"])
    mdl.download(
        domain,
        restrictions,
        mseed_storage=f"{savepath}/MSEED",
        stationxml_storage=f"{savepath}/StationXML"
    )

    logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
    logger.setLevel(logging.DEBUG)


def read_stationXML(signal_args):
    datadir = signal_args.datadir
    network_index = signal_args.network_index
    station_index = signal_args.station_index
    filepath = datadir + 'StationXML/'
    filespec = filepath + get_network(network_index) + '.' + \
               get_station(station_index) + '.xml'
    inv = read_inventory(filespec)
    return inv


def readStream_addBuffer(signal_args):
    datadir = signal_args.datadir
    network_index = signal_args.network_index
    station_index = signal_args.station_index
    channel_index = signal_args.channel_index
    datetime_index = signal_args.datetime_index
    taper_trace = signal_args.taper_trace
    pre_feed = signal_args.pre_feed
    # Define the taper and buffer lengths of the trace so that filtering does
    # not affect the stability of the data under analysis:
    buffer_front = taper_trace + pre_feed
    buffer_back = taper_trace
    # Define the time limits of the data.  Since the buffers will include data
    # from the previous and next days' data, include those dates into the search
    # range that will be used to load data into the script.
    t0 = get_datetime(datetime_index)
    t1 = t0 + timedelta(days=1)
    # t0 = get_datetime(datetime_index) + timedelta(hours=18)
    # t1 = t0 + timedelta(hours=1)
    time_start = t0 - timedelta(minutes=buffer_front)
    time_stop = t1 + timedelta(minutes=buffer_back)
    search_start = time_start.floor('D')
    search_stop = time_stop.ceil('D')
    search_range = pd.date_range(search_start, search_stop, freq='D')
    # Read in the trace from MSEED data:
    filepath = datadir + 'MSEED/'
    try:
        for i in range(len(search_range)-1):
            network = get_network(network_index)
            station = get_station(station_index)
            channel = get_channel(channel_index)
            filespec = network + '.' + station + '..' + channel + '__' + \
                       search_range[i].strftime('%Y%m%dT%H%M%S') + 'Z__' + \
                       search_range[i+1].strftime('%Y%m%dT%H%M%S') + 'Z.mseed'
            if i == 0:
                st = read(filepath + filespec)
            else:
                st += read(filepath + filespec)
    # Return (None, None, None) if files do not exist or can't be read:
    except:
        raise Exception('Unable to process station.')
        # return None, None, None

    # Merge the traces into one stream:
    st.merge(method=1, fill_value='interpolate', interpolation_samples=5)
    # Trim, detrend, and taper the data:
    tr = st[0].trim(
        starttime=UTCDateTime(time_start),
        endtime=UTCDateTime(time_stop)
    )
    # Return (None, None, None) if the data in the merged trace is a masked
    # array due to missing data points:
    if ma.is_masked(tr.data):
        raise Exception('Unable to process station.')
        # return None, None, None
    del st
    signal_args.time_start = time_start
    signal_args.time_stop = time_stop
    signal_args.buffer_front = buffer_front
    signal_args.buffer_back = buffer_back
    signal_args.network = network
    signal_args.station = station
    signal_args.channel = channel
    return tr, signal_args


def trace_processor(tr, inv, signal_args):
    taper_trace = signal_args.taper_trace
    time_start = signal_args.time_start
    time_stop = signal_args.time_stop
    cutoff = signal_args.cutoff
    tr.detrend(type='linear')
    tr.taper(max_percentage=0.5, type='hann', max_length=60*taper_trace)
    # Check sampling rate, re-sample to 100 Hz if necessary:
    fs = tr.stats.sampling_rate
    if fs == 200:
        tr.filter("lowpass", freq=50.0, corners=2, zerophase=True)
        tr.decimate(2, no_filter=True)
        fs = tr.stats.sampling_rate
    # Obtain statistics from trace:
    dt = tr.stats.delta
    npts = tr.stats.npts
    tvec = np.linspace(0, dt*(npts-1), npts)
    dtvec = pd.date_range(time_start, time_stop, freq=str(dt)+'S')
    # Remove instrument response from trace:
    pre_filt = (0.004, 0.01, (fs/2)*20, (fs/2)*40)
    tr.remove_response(inventory=inv, water_level=14, output='VEL',
                       pre_filt=pre_filt, plot=False)
    # Filter the trace:
    # tr.filter('highpass', freq=cutoff, corners=2, zerophase=True)
    tr.filter('bandpass', freqmin=cutoff, freqmax=30, zerophase=True)
    # signal_args.tvec = tvec
    signal_args.dtvec = dtvec
    signal_args.fs = fs
    return tr, signal_args


def workflow_wrapper(
        station_index,
        datadir,
        network_index,
        channel_index,
        datetime_index,
        taper_trace,
        pre_feed,
        cutoff,
        T_seg,
        NFFT,
        tpersnap,
        overlap,
        STA,
        LTA,
        trigger_on,
        trigger_off,
        debug=False
    ):
    signal_args = SigParam(
        datadir,
        network_index,
        station_index,
        channel_index,
        datetime_index,
        taper_trace,
        pre_feed,
        cutoff,
        T_seg,
        NFFT,
        tpersnap,
        overlap
    )
    detector_args = DetectorParam(STA, LTA, trigger_on, trigger_off)
    if debug:
        tr, signal_args = readStream_addBuffer(signal_args)
        inv = read_stationXML(signal_args)
        tr, signal_args = trace_processor(tr, inv, signal_args)
        tr_out, S_out, C_out, catdict = detector(
            tr,
            signal_args,
            detector_args,
            'classic'
        )
        return tr_out, S_out, C_out, catdict
    else:
        try:
            tr, signal_args = readStream_addBuffer(signal_args)
            inv = read_stationXML(signal_args)
            tr, signal_args = trace_processor(tr, inv, signal_args)
            tr_out, S_out, C_out, catdict = detector(
                tr,
                signal_args,
                detector_args,
                'classic'
            )
            return tr_out, S_out, C_out, catdict
        except:
            # print(f'\n    Station {get_station(station_index)} skipped.')
            return None, None, None, None


def KPDR_sac2mseed(datadir='.', destdir='.', response=False):
    print("Converting station KPDR SAC files to MSEED.")
    print(f"     Source: {datadir}\nDestination: {destdir}")
    if response:
        print("Processing WITH instrument response removal...")
        try:
            respf = [f for f in os.listdir(datadir) if 'RESP' in f][0]
        except IndexError:
            raise IndexError("No RESP files found in directory 'datadir'.")
    else:
        print("Processing WITHOUT instrument response removal...")

    files = sorted([f for f in os.listdir(datadir) if ('HDH' and 'SAC') in f])
    if len(files) < 3:
        raise ValueError("Not enough SAC files for continuous conversion.")

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    dt_start = file2dt(files[0]).date()
    dt_end = file2dt(files[-1]).date()
    dti = pd.date_range(dt_start, dt_end, freq='D')

    # NFFT = 4096
    overlap = 0.25
    taper_trace = 10
    pre_feed = 20
    cutoff = 0.4
    buffer_front = taper_trace + pre_feed
    buffer_back = taper_trace

    for d in tqdm(
        range(1, len(dti) - 1),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        unit="file",
    ):
        t0 = dti[d]
        t1 = dti[d+1]
        time_start = t0 - timedelta(minutes=buffer_front)
        time_stop = t1 + timedelta(minutes=buffer_back)
        search_start = time_start.floor('D')
        search_stop = time_stop.ceil('D')
        search_range = pd.date_range(search_start, search_stop,freq='D')

        first = True
        for i in range(len(search_range) - 1):
            flist = [f for f in files if file2dt(f).date() in search_range[0:-1]]
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
        try:
            tr.filter("lowpass", freq=0.4, corners=2, zerophase=True)
        except ValueError:
            raise ValueError("Check source files; missing data likely.")
        tr.decimate(100, no_filter=True)
        # Remove Instrument Response:
        if response:
            remove_trace(
                tr,
                f"{datadir}/{respf}",
                "DISP",
                paz_remove=None,
                pre_filt=(0.0015, 0.003, 0.5, 0.6),
                pitsasim=False,
                sacsim=True
            )
        tr.filter("bandpass", freqmin=0.001, freqmax=0.04, zerophase=True)
        tr.trim(starttime=UTCDateTime(t0), endtime=UTCDateTime(t1))
        destfname = f"KP.KPDR..HDH__{t0.strftime('%Y%m%dT%H%M%SZ')}__{t1.strftime('%Y%m%dT%H%M%SZ')}.mseed"
        tr.write(
            f"{destdir}/{destfname}",
            format="MSEED"
        )
    print("Complete.")


"""The following functions were written by Zhao Chen, UCSD
Compute and read instrument response
- read_file_response_text
- find_file_response_text
- read_file_response_function
- compute
- remove_trace
- remove_stream
"""
def read_file_response_text(file_response_text_name):
    """Read station information from the response file.

    Read station name, network name, location, channel, start time, end time,
    sensitivity and the corresponding frequency from the response file.

    Args:
        file_response_text_name (str): response file name.

    Returns:
        dict: dictionary containing the instrument information.

    Raises:
        ValueError: (1) If the information occurs multiple times in the response
            file. (2) If the decimation information is missed.

    """
    instrument_information = {
        'station': None, 'network': None, 'location': None, 'channel': None,
        't_start': None, 't_end': None, 'sensitivity': None,
        'frequency_sensitivity': None, 'sampling_rate': None
    }
    f_input = None
    decimation_factor = None
    with open(file_response_text_name) as file_response_text:
        response_text = file_response_text.read().split('\n')
        for i_line in range(len(response_text)):
            line = [item for item in response_text[i_line].split(' ') if item]
            if len(line) == 3 and line[1].lower() == 'station:':
                if not instrument_information['station']:
                    instrument_information['station'] = line[2]
                elif instrument_information['station'] != line[2]:
                    raise ValueError('Multiple station names!')
            elif len(line) == 3 and line[1].lower() == 'network:':
                if not instrument_information['network']:
                    instrument_information['network'] = line[2]
                elif instrument_information['network'] != line[2]:
                    raise ValueError('Multiple network names!')
            elif len(line) == 3 and line[1].lower() == 'location:':
                if not instrument_information['location']:
                    if line[2] == '??':
                        instrument_information['location'] = ''
                    else:
                        instrument_information['location'] = line[2]
                elif instrument_information['location'] != line[2]:
                    raise ValueError('Multiple location values!')
            elif len(line) == 3 and line[1].lower() == 'channel:':
                if not instrument_information['channel']:
                    instrument_information['channel'] = line[2]
                elif instrument_information['channel'] != line[2]:
                    raise ValueError('Multiple channels!')
            elif (len(line) == 4
                  and ' '.join(line[1:3]).lower() == 'start date:'):
                t_start = UTCDateTime().strptime(line[3], '%Y,%j,%H:%M:%S')
                if not instrument_information['t_start']:
                    instrument_information['t_start'] = t_start
                elif instrument_information['t_start'] != t_start:
                    raise ValueError('Multiple start dates!')
            elif (len(line) == 4
                  and ' '.join(line[1:3]).lower() == 'end date:'):
                t_end = UTCDateTime().strptime(line[3], '%Y,%j,%H:%M:%S')
                if not instrument_information['t_end']:
                    instrument_information['t_end'] = t_end
                elif instrument_information['t_end'] != t_end:
                    raise ValueError('Multiple end dates!')
            elif (len(line) == 5
                  and ' '.join(line[1:4]).lower() == 'input sample rate:'):
                f_input = float(line[4])
            elif (len(line) == 6
                  and ' '.join(line[1:5]).lower() == 'input sample rate (hz):'):
                f_input = float(line[5])
            elif (len(line) == 4
                  and ' '.join(line[1:3]).lower() == 'decimation factor:'):
                decimation_factor = float(line[3])
            elif len(line) == 3 and line[1].lower() == 'sensitivity:':
                sensitivity = float(line[2])
                if not instrument_information['sensitivity']:
                    instrument_information['sensitivity'] = sensitivity
                elif instrument_information['sensitivity'] != sensitivity:
                    raise ValueError('Multiple sensitivity values!')
            elif ((len(line) == 5 or len(line) == 6)
                  and ' '.join(line[1:4]).lower()
                      == 'frequency of sensitivity:'):
                frequency_sensitivity = float(line[4])
                if not instrument_information['frequency_sensitivity']:
                    instrument_information['frequency_sensitivity'] = (
                        frequency_sensitivity
                    )
                elif (instrument_information['frequency_sensitivity'] !=
                      frequency_sensitivity):
                    raise ValueError(
                        'Multiple frequency of sensitivity values!')
            else:
                continue
    if (f_input is not None) and (decimation_factor is not None):
        instrument_information['sampling_rate'] = f_input/decimation_factor
    else:
        raise ValueError('Decimation information missing!')
    return instrument_information


def find_file_response_text(tr, file_response_text_name_list):
    """Find the right response file corresponding to the trace.

    Args:
        tr (obspy.core.trace.Trace): data.
        file_response_text_name_list (list): response file name list.

    Returns:
        file_response_text_name (str): response file name.

    """
    for file_response_text_name in file_response_text_name_list:
        instrument_information = read_file_response_text(
            file_response_text_name)
        if (tr.stats.network == instrument_information['network'] and
                tr.stats.station == instrument_information['station'] and
                tr.stats.location == instrument_information['location'] and
                tr.stats.channel == instrument_information['channel']):
            return file_response_text_name
    raise ValueError('No corresponding response text file found!')


def read_file_response_function(file_response_function_name, n_f=1000):
    """Read instrument response function generated by evalresp.

    Args:
        file_response_function_name (str): name of the response function file
            generated by evalresp.

    Returns:
        numpy.ndarray: frequency
        numpy.ndarray: amplitude/phase response

    """
    # f = np.empty(n_f)
    # value = np.empty(n_f)
    f = []
    value = []
    with open(file_response_function_name) as file_response_function:
        response_function = file_response_function.read().split('\n')
        # for i_f in range(n_f):
        #     line = response_function[i_f].split(' ')
        #     f[i_f] = float(line[0])
        #     value[i_f] = float(line[1])
        for i_f, line in enumerate(response_function):
            line_break = line.split(' ')
            if len(line_break) < 2:
                continue
            f.append(float(line_break[0]))
            value.append(float(line_break[1]))
    return np.array(f), np.array(value)


def compute(file_response_text_name, f_min, f_max, n_f):
    """Compute response function by calling evalresp.

    Args:
        file_response_text_name (str): response file name.
        f_min (float): minimum frequency.
        f_max (float): maximum frequency.
        n_f (int): number of frequency samples.

    Returns:
        dict: dictionary containing the instrument information.

    """
    # Read instrument information from the response text file
    instrument_information = read_file_response_text(
        file_response_text_name)

    # Compute instrument response with evalresp
    run(['evalresp',
         instrument_information['station'], instrument_information['channel'],
         str(instrument_information['t_start'].year),
         str(instrument_information['t_start'].julday),
         str(f_min), str(f_max), str(n_f),
         '-f', file_response_text_name,
         '-t', instrument_information['t_start'].strftime('%H:%M:%S'),
         '-s', 'log'])

    # Response function file name generated by evalresp
    file_response_function_name_suffix = (
        instrument_information['network']
        + '.'
        + instrument_information['station']
        + '.'
        + instrument_information['location']
        + '.'
        + instrument_information['channel'])
    file_response_function_amplitude_name = (
        'AMP.' + file_response_function_name_suffix)
    file_response_function_phase_name = (
        'PHASE.' + file_response_function_name_suffix)

    return (instrument_information,
            file_response_function_amplitude_name,
            file_response_function_phase_name)


def remove_trace(tr,
                 file_response_text_name,
                 units,
                 taper_half_width=None,
                 **kwargs):
    """Remove instrument response for a single trace

    Args:
        tr (obspy.core.trace.Trace): data.
        file_response_text_name: response text file name, with path if not in
            './'.
        units (st): Output units. One of:
            ``"DISP"``
                displacement, output unit is meters
            ``"VEL"``
                velocity, output unit is meters/second
            ``"ACC"``
                acceleration, output unit is meters/second**2
        taper_half_width (int or float): half taper width (s)
        kwargs: arguments of obspy.core.trace.Trace.simulate.

    """
    seedresp = {'filename': file_response_text_name,
                'units': units[0:3].upper()}
    if taper_half_width:
        taper_fraction = (2 * taper_half_width
                          / (tr.stats.endtime - tr.stats.starttime))
        tr.simulate(seedresp=seedresp, taper_fraction=taper_fraction, **kwargs)
    else:
        tr.simulate(seedresp=seedresp, **kwargs)


def remove_stream(st,
                  file_response_text_name_list,
                  units,
                  taper_half_width=None,
                  **kwargs):
    """Remove instrument response for a data stream

    Args:
        st (obspy.core.stream.Stream): data.
        file_response_text_name_list (list): response text file name list,
        with path if not in the current folder.
        units (st): Output units. One of:
            ``"DISP"``
                displacement, output unit is meters
            ``"VEL"``
                velocity, output unit is meters/second
            ``"ACC"``
                acceleration, output unit is meters/second**2
        taper_half_width (int or float): half taper width (s)
        kwargs: arguments of obspy.core.trace.Trace.simulate.

    """
    st.merge()
    for i_trace in range(len(st)):
        file_response_text_name = find_file_response_text(
            st[i_trace], file_response_text_name_list)
        remove_trace(st[i_trace], file_response_text_name, units,
                     taper_half_width, **kwargs)
