from concurrent.futures import as_completed, ProcessPoolExecutor
import configparser
import csv
from datetime import datetime
from email.message import EmailMessage
import os
import pickle
import re
from shutil import copyfile
import smtplib
import ssl
import subprocess
import sys

from dotenv import load_dotenv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from twilio.rest import Client


class H5SeismicDataset(Dataset):
    """Loads samples from H5 dataset for use in native PyTorch dataloader."""
    def __init__(self, fname, transform=None):
        self.transform = transform
        self.fname = fname

    def __len__(self):
        m, _, _ = query_dbSize(self.fname)
        return m

    def __getitem__(self, idx):
        X = torch.from_numpy(read_h5(self.fname, idx))
        if self.transform:
            X = self.transform(X)
        return idx, X


class SeismoDataset(Dataset):
    "Converts ndarray already in memory to PyTorch dataset."
    def __init__(self, data, transform=None):
        self.data = torch.from_numpy(data).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.data)


class SpecgramShaper(object):
    """Crop & reshape data."""
    def __init__(self, n=None, o=None, transform='sample_norm_cent'):
        self.n = n
        self.o = o
        self.transform = transform

    def __call__(self, X):
        if self.n is not None and self.o is not None:
            N, O = X.shape
        else:
            X = X[:-1, 1:]
        if self.transform == "sample_norm":
            X /= np.abs(X).max(axis=(0,1))
        elif self.transform == "sample_norm_cent":
            # X = (X - X.mean(axis=(0,1))) / \
            # np.abs(X).max(axis=(0,1))
            X = (X - X.mean()) / np.abs(X).max()
        X = np.expand_dims(X, axis=0)
        return X


class SpecgramToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, X):
        return torch.from_numpy(X)


def calc_tuning_runs(hyperparameters):
    tuning_runs = 1
    for key in hyperparameters:
        tuning_runs *= len(hyperparameters[key])

    return(tuning_runs)


def config_training(universal, parameters, hyperparameters=None):
    config = configparser.ConfigParser()
    config['UNIVERSAL'] = universal
    config['PARAMETERS'] = parameters
    if hyperparameters is not None:
        config['HYPERPARAMETERS'] = hyperparameters
    fname = f"{universal['savepath']}/init_{parameters['mode']}.ini"
    with open(fname, 'w') as configfile:
        config.write(configfile)
    return fname


def distance_matrix(x, y, f):
    assert len(x) == len(y)
    M = len(x)
    dist = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            dist[i, j] = fractional_distance(
                x[np.newaxis,i],
                y[np.newaxis,j],
                f
            )
    return dist


def fractional_distance(x, y, f):
    diff = np.fabs(x - y) ** f
    dist = np.sum(diff, axis=1) ** (1 / f)
    return dist


def init_exp_env(mode, savepath, **kwargs):
    if mode == 'batch_predict':
        init_file = kwargs.get("init_file")
        exper = init_file.split("/")[-2][10:]
        serial_exp = exper[3:]
        run = f'Run{init_file.split("/")[-1][9:-4]}'
        savepath_exp = f'{savepath}/Trials/{exper}/{run}/'
    else:
        serial_exp = datetime.now().strftime('%Y%m%dT%H%M%S')
        if mode == 'pretrain':
            savepath_exp = f'{savepath}/Models/AEC/Exp{serial_exp}/'
        elif mode == 'train':
            savepath_exp = f'{savepath}/Models/DCM/Exp{serial_exp}/'
        elif mode == 'predict':
            savepath_exp = f'{savepath}/Trials/Exp{serial_exp}/'
        else:
            raise ValueError(
                'Incorrect mode selected; choose "pretrain", "train", or "eval".'
            )
    if not os.path.exists(savepath_exp):
        os.makedirs(savepath_exp)
    print('New experiment file structure created at:\n'
          f'{savepath_exp}')

    return savepath_exp, serial_exp


def init_output_env(savepath, mode, **kwargs):
    serial_run = datetime.now().strftime('%Y%m%dT%H%M%S')
    if mode == 'pretrain':
        savepath_run = f'{savepath}Run' + \
                       f'_BatchSz={kwargs.get("batch_size")}' + \
                       f'_LR={kwargs.get("lr")}'
        if not os.path.exists(savepath_run):
            os.makedirs(savepath_run)
        savepath_chkpnt = f'{savepath_run}/tmp'
        if not os.path.exists(savepath_chkpnt):
            os.makedirs(savepath_chkpnt)
        return savepath_run, serial_run, savepath_chkpnt
    elif mode == 'train':
        savepath_run = f'{savepath}Run' + \
                       f'_Clusters={kwargs.get("n_clusters")}' + \
                       f'_BatchSz={kwargs.get("batch_size")}' + \
                       f'_LR={kwargs.get("lr")}' + \
                       f'_gamma={kwargs.get("gamma")}' + \
                       f'_tol={kwargs.get("tol")}'
        return savepath_run, serial_run
    elif mode == 'predict':
        n_clusters = kwargs.get('n_clusters')
        with open(f'{savepath}{n_clusters}_Clusters', 'w') as f:
            pass
        savepath_run = []
        for label in range(n_clusters):
            savepath_cluster = f'{savepath}Cluster{label:02d}'
            if not os.path.exists(savepath_cluster):
                os.makedirs(savepath_cluster)
            savepath_run.append(savepath_cluster)
        return savepath_run, serial_run
    else:
        raise ValueError(
                'Incorrect mode selected; choose "pretrain", "train", or "eval".'
            )


# def load_dataset(
#         fname_dataset,
#         index,
#         send_message=False,
#         transform=None,
#         **kwargs
#     ):
#     '''
#     Arguments:
#       fname_dataset: Path to h5 dataset
#       index: List of indices to load
#       send_message: Boolean
#       transform: Data transformation (default: None, pixelwise, sample_norm, sample_norm_cent, sample_std)
#     '''
#     M = len(index)
#     if 'notqdm' in kwargs:
#         notqdm = kwargs.get("notqdm")
#     else:
#         notqdm = False
#
#     with h5py.File(fname_dataset, 'r') as f:
#         #samples, frequency bins, time bins, amplitude
#         DataSpec = '/4.0/Spectrogram'
#         dset = f[DataSpec]
#         m, n, o = dset.shape
#         m -= 1
#         if not notqdm:
#             print('-' * 80)
#             print(f'H5 file has {m} samples, {n} frequency bins, {o} time bins.')
#             print(f'Loading {M} samples...')
#         tic = datetime.now()
#
#         np.seterr(all='ignore')
#         # X = np.empty([M, n-2, o-173, 1])
#         # X = np.zeros([M, 1, 65, 175])
#         X = np.zeros([M, 69, 175])
#         idx_sample = np.empty([M,], dtype=np.int)
#         dset_arr = np.zeros([n, o])
#         count = 0
#         for i in tqdm(
#             range(M),
#             bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
#             disable=notqdm
#         ):
#             dset_arr = dset[index[i], :-1, 12:-14] # <---- This by itself doesn't work.
#             if transform == "sample_norm":
#                 dset_arr /= np.abs(dset_arr).max() # <---- This one works
#             elif transform == "sample_norm_cent":
#                 dset_arr = (dset_arr - dset_arr.mean()) / np.abs(dset_arr).max() # <---- This one works
#             elif transform == "sample_std":
#                 dset_arr = (dset_arr - dset_arr.mean()) / dset_arr.std() # <---- This one throws NaNs for loss in pre-training
#
#             X[count,:,:] = dset_arr
#             # X[count,:,:,:] = np.expand_dims(dset_arr, axis=0)
#             idx_sample[count,] = int(index[i])
#             count += 1
#
#         if transform == "pixelwise":
#             X = (X - X.mean(axis=0)) / np.abs(X).max(axis=0)
#
#         X = np.expand_dims(X, axis=1)
#
#         toc = datetime.now()
#         msgcontent = f'{M} spectrograms loaded successfully at {toc}.' + \
#                      f'\nTime Elapsed = {(toc-tic)}'
#         if not notqdm:
#             print(msgcontent)
#         if send_message:
#             msgsubj = 'Data Loaded'
#             notify(msgsubj, msgcontent)
#
#     return SeismoDataset(X)


def load_images(fname_dataset, index):
    with h5py.File(fname_dataset, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/4.0/Spectrogram'
        dset = f[DataSpec]
        X = np.zeros((len(index), 88, 101))
        for i, index in enumerate(index):
            dset_arr = dset[index, :, :]
            X[i] = dset_arr
    #         X = dset_arr / np.abs(dset_arr).max()
    #         X = (dset_arr - dset_arr.mean()) / dset_arr.std()
        fvec = dset[1, 0:86, 0]
        tvec = dset[1, 87, 1:]
    X = X[:, :-1, 1:]
    # tvec = tvec[12:-14]
    # fvec = fvec[:-1]

    X = (X - X.mean(axis=(1,2))[:,None,None]) / \
        np.abs(X).max(axis=(1,2))[:,None,None]

    X = np.expand_dims(X, axis=1)
    return X, tvec, fvec


def load_labels(exppath):
    csv_file = [f for f in os.listdir(exppath) if f.endswith('.csv')][0]
    csv_file = f'{exppath}/{csv_file}'
    data = np.genfromtxt(csv_file, delimiter=',')
    data = np.delete(data,0,0)
    data = data.astype('int')
    label = data[:,0]
    index = data[:,1]
    label_list = np.unique(label)
    return label, index, label_list


def load_weights(model, fname, device):
    model.load_state_dict(torch.load(fname, map_location=device), strict=False)
    model.eval()
    print(f'Weights loaded to {device}')
    return model


def make_dir(savepath_new, savepath_run="."):
    path = f"{savepath_run}/{savepath_new}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def make_exp(exppath, **kwargs):
    serial_exp = datetime.now().strftime('%Y%m%dT%H%M%S')
    savepath_exp = f"{exppath}/{serial_exp}"
    savepath_AEC = f"{savepath_exp}/AEC"
    savepath_DCM = f"{savepath_exp}/DCM"
    if not os.path.exists(savepath_exp):
        os.makedirs(savepath_exp)
    return savepath_exp, serial_exp


def make_pred_configs_batch(loadpath, savepath, overwrite=False):
    exper = loadpath.split("/")[-1]
    savepath = f"{savepath}/BatchEval_{exper}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    config_ = configparser.ConfigParser()
    tmp = [f for f in os.listdir(loadpath) if f.endswith('.ini')][0]
    config_.read(f"{loadpath}/{tmp}")
    transform = config_['PARAMETERS']['transform']

    count_wr = 0
    count_ow = 0
    count_sk = 0
    runlist = [f for f in os.listdir(f'{loadpath}') if "Run" in f]
    for run in runlist:
        fname = f'{savepath}/init_pred_{run[4:]}.ini'
        if os.path.isfile(fname):
            if not overwrite:
                count_sk += 1
                continue
            elif overwrite:
                count_ow += 1
        else:
            count_wr += 1

        config = configparser.ConfigParser()
        config['UNIVERSAL'] = {
            'mode': 'predict',
            'fname_dataset': '../../../Data/DetectionData_4s.h5',
            'savepath': '../../../Outputs/',
            'indexpath': '../../../Data/TraValIndex_M=125000_Res=0.0_20200828T005531.pkl'
        }
        saved_weights = [f for f in os.listdir(f'{loadpath}/{run}') if f.endswith('.pt')][0]
        config['PARAMETERS'] = {
            'M': 'all',
            'exclude': 'False',
            'batch_size': '1024',
            'show': 'False',
            'send_message': 'False',
            'max_workers': '14',
            'n_clusters': parse_nclusters(run),
            'saved_weights': f'{loadpath}/{run}/{saved_weights}',
            'transform': transform
        }
        with open(fname, 'w') as configfile:
            config.write(configfile)

    print(f'Config Files: {count_wr} written, {count_ow} overwritten, {count_sk} skipped.')
    return savepath


# def multi_load(path, index, send_message=False, transform=None, **kwargs):
#     '''
#     Arguments:
#       fname_dataset: Path to h5 dataset
#       index: List of indices to load
#       send_message: Boolean
#       transform: Data transformation (default: None, pixelwise, sample_norm, sample_norm_cent, sample_std)
#     '''
#     M = len(index)
#     if 'notqdm' in kwargs:
#         notqdm = kwargs.get("notqdm")
#     else:
#         notqdm = False
#         m, n, o = query_dbSize(path)
#         print('--------------------------------------------------------------')
#         print(f'H5 file has {m} samples, {n} frequency bins, {o} time bins.')
#         print(f'Loading {M} samples...')
#     tic = datetime.now()
#     A = [
#             {
#                 'fname_dataset': path,
#                 'index': index[i],
#             } for i in range(M)]
#     X = np.zeros((M, 88, 101))
#     with ProcessPoolExecutor(max_workers=16) as exec:
#         futures = [exec.submit(read_h5, **a) for a in A]
#         kwargs = {
#             'total': int(len(futures)),
#             'unit': 'samples',
#             'unit_scale': True,
#             'bar_format': '{l_bar}{bar:20}{r_bar}{bar:-20b}',
#             'leave': True,
#             'disable': notqdm
#         }
#         for i, future in enumerate(tqdm(as_completed(futures), **kwargs)):
#             X[i, :, :] = future.result()
#
#     if not notqdm:
#         print("Transforming data...", end="", flush=True)
#     X = X[:, :-1, 1:]
#     if transform == "sample_norm": # <-------------------------- Works
#         X /= np.abs(X).max(axis=(1,2))[:,None,None]
#     elif transform == "sample_norm_cent": # <------------------- Works
#         X = (X - X.mean(axis=(1,2))[:,None,None]) / \
#             np.abs(X).max(axis=(1,2))[:,None,None]
#     elif transform == "sample_std": # <------------------------- Doesn't work
#         X = (X - X.mean(axis=(1,2))[:,None,None]) / \
#             X.std(axis=(1,2))[:,None,None]
#     elif transform == "pixelwise":
#         X = (X - X.mean(axis=0)) / np.abs(X).max(axis=0)
#     X = np.expand_dims(X, axis=1)
#     if not notqdm:
#         print('complete.')
#
#     toc = datetime.now()
#     msgcontent = f'{M} spectrograms loaded successfully at {toc}.' + \
#                  f'\nTime Elapsed = {(toc-tic)}'
#     if not notqdm:
#         print(msgcontent)
#     if send_message:
#         msgsubj = 'Data Loaded'
#         notify(msgsubj, msgcontent)
#     return SeismoDataset(X)


def measure_class_inertia(data, centroids, n_clusters):
    inertia = np.empty(n_clusters)
    for j in range(n_clusters):
        mu = centroids[j]
        inertia[j] = np.sum(np.sqrt(np.sum((data - mu) ** 2, axis=1)) ** 2)
    return inertia


def notify(msgsubj, msgcontent):
    '''Written by William Jenkins, 19 June 2020, wjenkins@ucsd.edu3456789012
    Scripps Institution of Oceanography, UC San Diego
    This function uses the SMTP and Twilio APIs to send an email and WhatsApp
    message to a user defined in environmental variables stored in a .env file
    within the same directory as this module.  Sender credentials are stored
    similarly.'''
    load_dotenv()
    msg = EmailMessage()
    msg['Subject'] = msgsubj
    msg.set_content(msgcontent)
    username = os.getenv('ORIG_USERNAME')
    password = os.getenv('ORIG_PWD')
    try:
        # Create a secure SSL context
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(
                'smtp.gmail.com',
                port=465,
                context=context
            ) as s:
            s.login(username, password)
            receiver_email = os.getenv('RX_EMAIL')
            s.sendmail(username, receiver_email, msg.as_string())
            print('Job completion notification sent by email.')
    except:
        print('Unable to send email notification upon job completion.')
        pass
    try:
        client = Client()
        orig_whatsapp_number = 'whatsapp:' + os.getenv('ORIG_PHONE_NUMBER')
        rx_whatsapp_number = 'whatsapp:' + os.getenv('RX_PHONE_NUMBER')
        msgcontent = f'*{msgsubj}*\n{msgcontent}'
        client.messages.create(
            body=msgcontent,
            from_=orig_whatsapp_number,
            to=rx_whatsapp_number
        )
        print('Job completion notification sent by WhatsApp.')
    except:
        print('Unable to send WhatsApp notification upon job completion.')
        pass


def parse_nclusters(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """
    rx_dict = {'n_clusters': re.compile(r'Clusters=(?P<n_clusters>\d+)')}
    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return match.group('n_clusters')
        else:
            raise Exception('Unable to parse filename for n_clusters.')


def query_dbSize(path):
    with h5py.File(path, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/4.0/Spectrogram'
        dset = f[DataSpec]
        m, n, o = dset.shape
        return m, n, o


def read_h5(fname_dataset, index):
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/4.0/Spectrogram'
        return f[DataSpec][index]


def save_exp_config(savepath, serial, init_file, parameters, hyperparameters):
    fname = f'{savepath}ExpConfig{serial}'
    if hyperparameters is not None:
        configs = [parameters, hyperparameters]
    else:
        configs = parameters
    copyfile(init_file, f'{fname}.ini')
    with open(f'{fname}.txt', 'w') as f:
        f.write(str(configs))
    with open(f'{fname}.pkl', 'wb') as f:
        pickle.dump(configs, f)


def save_history(training_history, validation_history, savepath, run_serial):
    if validation_history is not None:
        fname = f'{savepath}/AEC_History{run_serial}.csv'
        d1 = training_history.copy()
        d2 = validation_history.copy()
        modes = ['training', 'validation']
        for mode in range(len(modes)):
            if modes[mode] == 'training':
                d = d1
            elif modes[mode] == 'validation':
                d = d2
            for j in range(2):
                newkey = '{}_{}'.format(modes[mode], list(d.keys())[0])
                oldkey = list(d.keys())[0]
                d[newkey] = d.pop(oldkey)
        d2.update(d1)
        del d1
    else:
        fname = f'{savepath}/DCM_History{run_serial}.csv'
        d2 = training_history

    with open(fname, 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(d2.keys())
        w.writerows(zip(*d2.values()))


def save_labels(label_list, savepath, serial=None):
    if serial is not None:
        fname = f'{savepath}/Labels{serial}.csv'
    else:
        fname = f'{savepath}/Labels.csv'
    keys = label_list[0].keys()
    if not os.path.exists(fname):
        with open(fname, 'w') as csvfile:
            w = csv.DictWriter(csvfile, keys)
            w.writeheader()
            w.writerows(label_list)
    else:
        with open(fname, 'a') as csvfile:
            w = csv.DictWriter(csvfile, keys)
            w.writerows(label_list)


def set_device(cuda_device=None):
    if torch.cuda.is_available and (cuda_device is not None):
        device = torch.device(f'cuda:{cuda_device}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA device available, using GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA device not available, using CPU.')
    return device


def start_tensorboard(logdir, tbport):
    cmd = f"python -m tensorboard.main --logdir=. --port={tbport} --samples_per_plugin images=1000"
    p = subprocess.Popen(cmd, cwd=logdir, shell=True)
    tbpid = p.pid
    print(f"Tensorboard server available at http://localhost:{tbport}; PID={tbpid}")
    return tbpid


# =============================================================================
#  Functions to set/save/get indices of training/validation/prediction samples.
# =============================================================================
def set_TraVal_index(M, fname_dataset, reserve=0.0):
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/4.0/Spectrogram'
        m, _, _ = f[DataSpec].shape
        if M > m:
            print(f'{M} spectrograms requested, but only {m} '
                  f'available in database; setting M to {m}.')
            M = m
    index = np.random.choice(
        np.arange(1,m),
        size=int(M * (1+reserve)),
        replace=False
    )
    split_fraction = 0.8
    split = int(split_fraction * len(index))
    index_tra = index[0:split]
    index_val = index[split:]
    return index_tra, index_val, M


def save_TraVal_index(M, fname_dataset, savepath, reserve=0.0):
    index_tra, index_val, M = set_TraVal_index(M, fname_dataset)
    index = dict(
        index_tra=index_tra,
        index_val=index_val
    )
    serial = datetime.now().strftime('%Y%m%dT%H%M%S')
    # savepath = f'{savepath}TraValIndex_M={M}_Res={reserve}_{serial}.pkl'
    savepath = f'{savepath}/TraValIndex_M={M}.pkl'
    with open(savepath, 'wb') as f:
        pickle.dump(index, f)
    print(f'{M} training & validation indices saved to:')
    print(savepath)
    return index_tra, index_val, savepath


def load_TraVal_index(fname_dataset, loadpath):
    with open(loadpath, 'rb') as f:
        data = pickle.load(f)
        index_tra = data['index_tra']
        index_val = data['index_val']
    return index_tra, index_val


def set_Tst_index(M, fname_dataset, indexpath, reserve=0.0, exclude=True):
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/4.0/Spectrogram'
        m, _, _ = f[DataSpec].shape
    index = np.arange(0, m)
    if exclude:
        idx_ex = load_TraVal_index(fname_dataset, indexpath)
        idx_ex = sorted(set(np.concatenate(idx_ex).flatten()))
        index_avail = np.setdiff1d(index, idx_ex)
    else:
        index_avail = index
    index_tst = np.random.choice(
        index_avail,
        size=int(M * (1+reserve)),
        replace=False
    )
    return index_tst


def set_M(fname_dataset, indexpath, exclude=True):
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/4.0/Spectrogram'
        m, _, _ = f[DataSpec].shape
    print('Determining number of prediction samples...')
    print(f'{m} samples in dataset...')
    if exclude:
        idx_tra, idx_val = load_TraVal_index(fname_dataset, indexpath)
        M_TraVal = len(idx_tra) + len(idx_val)
        print(f'{M_TraVal} training/validation samples...')
        M = m - M_TraVal
        print(f'{m} - {M_TraVal} = {M} prediction samples to be used.')
    else:
        M = m
        print(f'{M} prediction samples to be used.')
    return M
