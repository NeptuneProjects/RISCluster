from datetime import datetime, timedelta
import glob
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
from scipy.io import loadmat
from tqdm import tqdm


# class EnvironmentCatalogue(object):
#     def __init__(self, station, aws, path):
#         self.station = station
#         self.aws = aws
#         self.path = path
#         self.df = self.build_df(self.station, self.aws, self.path)
#
#
#     def build_df(self, station, aws, path):
#         # Load tidal data:
#         sta_ind = get_station(station)
#         if station == "RS08" or station == "RS11":
#             sta_ind -= 1
#         elif station == "RS09":
#             sta_ind += 1
#         elif station == "RS17":
#             sta_ind -= 2
#         data = loadmat(f"{path}/RIS_Tides.mat")["z"][sta_ind,:]
#         df_tide = pd.DataFrame(data={"tide": data}, index=pd.date_range("2014-12-01", "2016-12-01", freq="10min"), columns=["tide"])
#         # Load sea ice concentration:
#         data = loadmat(f"{path}/NSIDC-0051.mat")
#         df_ice = pd.DataFrame(data={"sea_ice_conc": data["C"].squeeze()*100}, index=pd.to_datetime(data["date"]), columns=['sea_ice_conc'])
#         # Load meteo data:
#         df_meteo = read_meteo(f"{path}/RIS_Meteo/{aws}*.txt")
#         # Load ERA5 data:
#         df_energy = read_ERA5(f"{path}/SDM_jan2016_ERA5.csv")
#         # Load Wave Amplitude data:
#         df_wave = read_KPDR(f"{path}/RIS_Seismic/KPDR_0.001_0.04.mat")
#
#         # Combine datasets into one dataframe:
#         df = pd.concat([df_tide, df_ice, df_meteo, df_energy, df_wave], axis=1)
#         df["sea_ice_conc"] = df["sea_ice_conc"].interpolate()
#         df["net_sfc_melt_energy"] = df["net_sfc_melt_energy"].interpolate()
#         return df


def _copy_attributes(in_object, out_object):
    '''Copy attributes between 2 HDF5 objects.'''
    for key, value in in_object.attrs.items():
        out_object.attrs[key] = value


def _find_indeces(index, source, stations):
    with h5py.File(source, 'r') as f:
        metadata = json.loads(f['/4.0/Catalogue'][index])
    if metadata["Station"] in stations:
        return index
    else:
        return np.nan


# def file2dt(fname):
#     fname = fname.split('.')[0:5]
#     dt = datetime.strptime(
#         f'{fname[0]} {fname[1]} {fname[2]} {fname[3]} {fname[4]}',
#         '%Y %j %H %M %S'
#     )
#     return dt


def get_channel(channel_index):
    '''Input: Integer channel index (0-2).
       Output: Channel name (str)'''
    channel_list = ['HHE', 'HHN', 'HHZ']
    channel_name = channel_list[channel_index]
    return channel_name


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
        DataSpec = '/4.0/Catalogue'
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


def get_station(station):
    '''Returns station index or station name, depending on whether input is a
    name (string) or index (integer).

    Parameters
    ----------
    station : str, int
        Station name (str), Station index (int)

    Returns
    -------
    station: int, str
        Station index (int), Station name (str)
    '''
    station_list = ['DR01', 'DR02', 'DR03', 'DR04', 'DR05', 'DR06', 'DR07',
                   'DR08', 'DR09', 'DR10', 'DR11', 'DR12', 'DR13', 'DR14',
                   'DR15', 'DR16', 'RS01', 'RS02', 'RS03', 'RS04', 'RS05',
                   'RS06', 'RS07', 'RS08', 'RS09', 'RS10', 'RS11', 'RS12',
                   'RS13', 'RS14', 'RS15', 'RS16', 'RS17', 'RS18']
    if isinstance(station, int):
        return station_list[station]
    elif isinstance(station, str):
        return station_list.index(station)


# def read_ERA5(path):
#     '''Reads ERA5 data from .csv file to Pandas dataframe.
#
#     Parameters
#     ----------
#     path : str
#         Path to ERA5 files
#
#     Returns
#     -------
#     dataframe : Pandas dataframe
#         Dataframe whose index is datetime, and whose columns are net surface
#         melting energy (units).
#
#     Notes
#     -----
#     Antarctica AWS data accessed from
#     https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5.
#     '''
#     file_list = glob.glob(path)
#     first = True
#     for file in file_list:
#         df = pd.read_csv(
#             file,
#             index_col=[0],
#             usecols=["time","net_sfc_melt_energy"],
#             parse_dates=True,
#             infer_datetime_format=True
#         )
#         if first:
#             df_energy = df
#             first = False
#         else:
#             df_energy = df_energy.append(df)
#     return df_energy
#
#
# def read_KPDR(path):
#     data = loadmat("/Users/williamjenkins/Research/Workflows/RIS_Clustering/Data/KPDR_0.001_0.04.mat")
#     datenums = data["t"].squeeze()
#     timestamps = pd.to_datetime(datenums-719529, unit='D').round("S")
#     ampl = data["a"].squeeze()
#     df_wave = pd.DataFrame(data={"wave_ampl": ampl}, index=timestamps).resample("10T").interpolate()
#     return df_wave
#
#
# def read_meteo(path):
#     '''Reads AWS data from tab-separated .txt file to Pandas dataframe.
#
#     Parameters
#     ----------
#     path : str
#         Path to AWS files
#
#     Returns
#     -------
#     dataframe : Pandas dataframe
#         Dataframe whose index is datetime, and whose columns are temperature
#         (C) and wind speed (m/s).
#
#     Notes
#     -----
#     Antarctica AWS data accessed from https://amrc.ssec.wisc.edu.
#     '''
#     file_list = glob.glob(path)
#     first = True
#     for file in file_list:
#         df = pd.read_csv(
#             file,
#             sep=" ",
#             header=0,
#             names=["Year","Month","Day","Time","temp","wind_spd"],
#             usecols=[0, 2, 3, 4, 5, 7],
#             dtype={"Year": str, "Month": str, "Day": str, "Time": str},
#             skipinitialspace=True,
#             skiprows=1,
#             na_values=444.0
#         )
#         df["Hour"] = df.Time.str.slice(0, 2)
#         df["Minute"] = df.Time.str.slice(2, 4)
#         dti = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
#         df.drop(columns=["Year", "Month", "Day", "Time", "Hour", "Minute"], inplace=True)
#         df.index = dti
#         if first:
#             df_meteo = df
#             first = False
#         else:
#             df_meteo = df_meteo.append(df)
#     return df_meteo.sort_index()[datetime(2014,12,1):datetime(2016,12,1)]
#
#
# def KPDR_sac2mseed(datadir='.', destdir='.', response=False):
#     print("Converting station KPDR SAC files to MSEED.")
#     print(f"     Source: {datadir}\nDestination: {destdir}")
#     if response:
#         print("Processing WITH instrument response removal...")
#         try:
#             respf = [f for f in os.listdir(datadir) if 'RESP' in f][0]
#         except IndexError:
#             raise IndexError("No RESP files found in directory 'datadir'.")
#     else:
#         print("Processing WITHOUT instrument response removal...")
#
#     files = sorted([f for f in os.listdir(datadir) if ('HDH' and 'SAC') in f])
#     if len(files) < 3:
#         raise ValueError("Not enough SAC files for continuous conversion.")
#
#     if not os.path.exists(destdir):
#         os.makedirs(destdir)
#
#     dt_start = file2dt(files[0]).date()
#     dt_end = file2dt(files[-1]).date()
#     dti = pd.date_range(dt_start, dt_end, freq='D')
#
#     # NFFT = 4096
#     overlap = 0.25
#     taper_trace = 10
#     pre_feed = 20
#     cutoff = 0.4
#     buffer_front = taper_trace + pre_feed
#     buffer_back = taper_trace
#
#     for d in tqdm(
#         range(1, len(dti) - 1),
#         bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
#         unit="file",
#     ):
#         t0 = dti[d]
#         t1 = dti[d+1]
#         time_start = t0 - timedelta(minutes=buffer_front)
#         time_stop = t1 + timedelta(minutes=buffer_back)
#         search_start = time_start.floor('D')
#         search_stop = time_stop.ceil('D')
#         search_range = pd.date_range(search_start, search_stop,freq='D')
#
#         first = True
#         for i in range(len(search_range) - 1):
#             flist = [f for f in files if file2dt(f).date() in search_range[0:-1]]
#             for f, fname in enumerate(flist):
#                 if first:
#                     st = read(f"{datadir}/{fname}")
#                     first = False
#                 else:
#                     st += read(f"{datadir}/{fname}")
#         st.merge(method=1, fill_value='interpolate', interpolation_samples=5)
#         tr = st[0].trim(starttime=UTCDateTime(time_start), endtime=UTCDateTime(time_stop))
#
#         tr.detrend(type='linear')
#         tr.taper(max_percentage=0.5, type='hann', max_length=60*taper_trace)
#         fs = tr.stats.sampling_rate
#         # Decimate:
#         try:
#             tr.filter("lowpass", freq=0.4, corners=2, zerophase=True)
#         except ValueError:
#             raise ValueError("Check source files; missing data likely.")
#         tr.decimate(100, no_filter=True)
#         # Remove Instrument Response:
#         if response:
#             remove_trace(
#                 tr,
#                 f"{datadir}/{respf}",
#                 "DISP",
#                 paz_remove=None,
#                 pre_filt=(0.0015, 0.003, 0.5, 0.6),
#                 pitsasim=False,
#                 sacsim=True
#             )
#         tr.filter("bandpass", freqmin=0.001, freqmax=0.04, zerophase=True)
#         tr.trim(starttime=UTCDateTime(t0), endtime=UTCDateTime(t1))
#         destfname = f"KP.KPDR..HDH__{t0.strftime('%Y%m%dT%H%M%SZ')}__{t1.strftime('%Y%m%dT%H%M%SZ')}.mseed"
#         tr.write(
#             f"{destdir}/{destfname}",
#             format="MSEED"
#         )
#     print("Complete.")
#
#
# """The following functions were written by Zhao Chen, UCSD
# Compute and read instrument response
# - read_file_response_text
# - find_file_response_text
# - read_file_response_function
# - compute
# - remove_trace
# - remove_stream
# """
# def read_file_response_text(file_response_text_name):
#     """Read station information from the response file.
#
#     Read station name, network name, location, channel, start time, end time,
#     sensitivity and the corresponding frequency from the response file.
#
#     Args:
#         file_response_text_name (str): response file name.
#
#     Returns:
#         dict: dictionary containing the instrument information.
#
#     Raises:
#         ValueError: (1) If the information occurs multiple times in the response
#             file. (2) If the decimation information is missed.
#
#     """
#     instrument_information = {
#         'station': None, 'network': None, 'location': None, 'channel': None,
#         't_start': None, 't_end': None, 'sensitivity': None,
#         'frequency_sensitivity': None, 'sampling_rate': None
#     }
#     f_input = None
#     decimation_factor = None
#     with open(file_response_text_name) as file_response_text:
#         response_text = file_response_text.read().split('\n')
#         for i_line in range(len(response_text)):
#             line = [item for item in response_text[i_line].split(' ') if item]
#             if len(line) == 3 and line[1].lower() == 'station:':
#                 if not instrument_information['station']:
#                     instrument_information['station'] = line[2]
#                 elif instrument_information['station'] != line[2]:
#                     raise ValueError('Multiple station names!')
#             elif len(line) == 3 and line[1].lower() == 'network:':
#                 if not instrument_information['network']:
#                     instrument_information['network'] = line[2]
#                 elif instrument_information['network'] != line[2]:
#                     raise ValueError('Multiple network names!')
#             elif len(line) == 3 and line[1].lower() == 'location:':
#                 if not instrument_information['location']:
#                     if line[2] == '??':
#                         instrument_information['location'] = ''
#                     else:
#                         instrument_information['location'] = line[2]
#                 elif instrument_information['location'] != line[2]:
#                     raise ValueError('Multiple location values!')
#             elif len(line) == 3 and line[1].lower() == 'channel:':
#                 if not instrument_information['channel']:
#                     instrument_information['channel'] = line[2]
#                 elif instrument_information['channel'] != line[2]:
#                     raise ValueError('Multiple channels!')
#             elif (len(line) == 4
#                   and ' '.join(line[1:3]).lower() == 'start date:'):
#                 t_start = UTCDateTime().strptime(line[3], '%Y,%j,%H:%M:%S')
#                 if not instrument_information['t_start']:
#                     instrument_information['t_start'] = t_start
#                 elif instrument_information['t_start'] != t_start:
#                     raise ValueError('Multiple start dates!')
#             elif (len(line) == 4
#                   and ' '.join(line[1:3]).lower() == 'end date:'):
#                 t_end = UTCDateTime().strptime(line[3], '%Y,%j,%H:%M:%S')
#                 if not instrument_information['t_end']:
#                     instrument_information['t_end'] = t_end
#                 elif instrument_information['t_end'] != t_end:
#                     raise ValueError('Multiple end dates!')
#             elif (len(line) == 5
#                   and ' '.join(line[1:4]).lower() == 'input sample rate:'):
#                 f_input = float(line[4])
#             elif (len(line) == 6
#                   and ' '.join(line[1:5]).lower() == 'input sample rate (hz):'):
#                 f_input = float(line[5])
#             elif (len(line) == 4
#                   and ' '.join(line[1:3]).lower() == 'decimation factor:'):
#                 decimation_factor = float(line[3])
#             elif len(line) == 3 and line[1].lower() == 'sensitivity:':
#                 sensitivity = float(line[2])
#                 if not instrument_information['sensitivity']:
#                     instrument_information['sensitivity'] = sensitivity
#                 elif instrument_information['sensitivity'] != sensitivity:
#                     raise ValueError('Multiple sensitivity values!')
#             elif ((len(line) == 5 or len(line) == 6)
#                   and ' '.join(line[1:4]).lower()
#                       == 'frequency of sensitivity:'):
#                 frequency_sensitivity = float(line[4])
#                 if not instrument_information['frequency_sensitivity']:
#                     instrument_information['frequency_sensitivity'] = (
#                         frequency_sensitivity
#                     )
#                 elif (instrument_information['frequency_sensitivity'] !=
#                       frequency_sensitivity):
#                     raise ValueError(
#                         'Multiple frequency of sensitivity values!')
#             else:
#                 continue
#     if (f_input is not None) and (decimation_factor is not None):
#         instrument_information['sampling_rate'] = f_input/decimation_factor
#     else:
#         raise ValueError('Decimation information missing!')
#     return instrument_information
#
#
# def find_file_response_text(tr, file_response_text_name_list):
#     """Find the right response file corresponding to the trace.
#
#     Args:
#         tr (obspy.core.trace.Trace): data.
#         file_response_text_name_list (list): response file name list.
#
#     Returns:
#         file_response_text_name (str): response file name.
#
#     """
#     for file_response_text_name in file_response_text_name_list:
#         instrument_information = read_file_response_text(
#             file_response_text_name)
#         if (tr.stats.network == instrument_information['network'] and
#                 tr.stats.station == instrument_information['station'] and
#                 tr.stats.location == instrument_information['location'] and
#                 tr.stats.channel == instrument_information['channel']):
#             return file_response_text_name
#     raise ValueError('No corresponding response text file found!')
#
#
# def read_file_response_function(file_response_function_name, n_f=1000):
#     """Read instrument response function generated by evalresp.
#
#     Args:
#         file_response_function_name (str): name of the response function file
#             generated by evalresp.
#
#     Returns:
#         numpy.ndarray: frequency
#         numpy.ndarray: amplitude/phase response
#
#     """
#     # f = np.empty(n_f)
#     # value = np.empty(n_f)
#     f = []
#     value = []
#     with open(file_response_function_name) as file_response_function:
#         response_function = file_response_function.read().split('\n')
#         # for i_f in range(n_f):
#         #     line = response_function[i_f].split(' ')
#         #     f[i_f] = float(line[0])
#         #     value[i_f] = float(line[1])
#         for i_f, line in enumerate(response_function):
#             line_break = line.split(' ')
#             if len(line_break) < 2:
#                 continue
#             f.append(float(line_break[0]))
#             value.append(float(line_break[1]))
#     return np.array(f), np.array(value)
#
#
# def compute(file_response_text_name, f_min, f_max, n_f):
#     """Compute response function by calling evalresp.
#
#     Args:
#         file_response_text_name (str): response file name.
#         f_min (float): minimum frequency.
#         f_max (float): maximum frequency.
#         n_f (int): number of frequency samples.
#
#     Returns:
#         dict: dictionary containing the instrument information.
#
#     """
#     # Read instrument information from the response text file
#     instrument_information = read_file_response_text(
#         file_response_text_name)
#
#     # Compute instrument response with evalresp
#     run(['evalresp',
#          instrument_information['station'], instrument_information['channel'],
#          str(instrument_information['t_start'].year),
#          str(instrument_information['t_start'].julday),
#          str(f_min), str(f_max), str(n_f),
#          '-f', file_response_text_name,
#          '-t', instrument_information['t_start'].strftime('%H:%M:%S'),
#          '-s', 'log'])
#
#     # Response function file name generated by evalresp
#     file_response_function_name_suffix = (
#         instrument_information['network']
#         + '.'
#         + instrument_information['station']
#         + '.'
#         + instrument_information['location']
#         + '.'
#         + instrument_information['channel'])
#     file_response_function_amplitude_name = (
#         'AMP.' + file_response_function_name_suffix)
#     file_response_function_phase_name = (
#         'PHASE.' + file_response_function_name_suffix)
#
#     return (instrument_information,
#             file_response_function_amplitude_name,
#             file_response_function_phase_name)
#
#
# def remove_trace(tr,
#                  file_response_text_name,
#                  units,
#                  taper_half_width=None,
#                  **kwargs):
#     """Remove instrument response for a single trace
#
#     Args:
#         tr (obspy.core.trace.Trace): data.
#         file_response_text_name: response text file name, with path if not in
#             './'.
#         units (st): Output units. One of:
#             ``"DISP"``
#                 displacement, output unit is meters
#             ``"VEL"``
#                 velocity, output unit is meters/second
#             ``"ACC"``
#                 acceleration, output unit is meters/second**2
#         taper_half_width (int or float): half taper width (s)
#         kwargs: arguments of obspy.core.trace.Trace.simulate.
#
#     """
#     seedresp = {'filename': file_response_text_name,
#                 'units': units[0:3].upper()}
#     if taper_half_width:
#         taper_fraction = (2 * taper_half_width
#                           / (tr.stats.endtime - tr.stats.starttime))
#         tr.simulate(seedresp=seedresp, taper_fraction=taper_fraction, **kwargs)
#     else:
#         tr.simulate(seedresp=seedresp, **kwargs)
#
#
# def remove_stream(st,
#                   file_response_text_name_list,
#                   units,
#                   taper_half_width=None,
#                   **kwargs):
#     """Remove instrument response for a data stream
#
#     Args:
#         st (obspy.core.stream.Stream): data.
#         file_response_text_name_list (list): response text file name list,
#         with path if not in the current folder.
#         units (st): Output units. One of:
#             ``"DISP"``
#                 displacement, output unit is meters
#             ``"VEL"``
#                 velocity, output unit is meters/second
#             ``"ACC"``
#                 acceleration, output unit is meters/second**2
#         taper_half_width (int or float): half taper width (s)
#         kwargs: arguments of obspy.core.trace.Trace.simulate.
#
#     """
#     st.merge()
#     for i_trace in range(len(st)):
#         file_response_text_name = find_file_response_text(
#             st[i_trace], file_response_text_name_list)
#         remove_trace(st[i_trace], file_response_text_name, units,
#                      taper_half_width, **kwargs)
