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
import sys

from dotenv import load_dotenv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from twilio.rest import Client

class testDataset(Dataset):

    def __init__(self, fname_dataset, M, transform=None):
        self.transform = transform

    def __getitem__():
        pass

    def __len__():
        pass

class SeismoDataset(Dataset):

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

class SuppressStdout(object):

    def __init__(self, suppress=True):
        self.suppress = suppress
        self.sys_stdout_ref = None

    def __enter__(self):
        self.sys_stdout_ref = sys.stdout
        if self.suppress:
            sys.stdout = self
        return sys.stdout

    def __exit__(self, type, value, traceback):
        sys.stdout = self.sys_stdout_ref

    def write(self):
        pass

def calc_tuning_runs(hyperparameters):
    tuning_runs = 1
    for key in hyperparameters:
        tuning_runs *= len(hyperparameters[key])

    return(tuning_runs)

def init_exp_env(mode, savepath, **kwargs):
    if mode == 'batch_predict':
        init_file = kwargs.get("init_file")
        exper = init_file.split("/")[-2][10:]
        serial_exp = exper[3:]
        run = f'Run{init_file.split("/")[-1][9:-4]}'
        savepath_exp = f'{savepath}Trials/{exper}/{run}/'
    else:
        serial_exp = datetime.now().strftime('%Y%m%dT%H%M%S')
        if mode == 'pretrain':
            savepath_exp = f'{savepath}Models/AEC/Exp{serial_exp}/'
        elif mode == 'train':
            savepath_exp = f'{savepath}Models/DCM/Exp{serial_exp}/'
        elif mode == 'predict':
            savepath_exp = f'{savepath}Trials/Exp{serial_exp}/'
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
    elif mode == 'train':
        savepath_run = f'{savepath}Run' + \
                       f'_Clusters={kwargs.get("n_clusters")}' + \
                       f'_BatchSz={kwargs.get("batch_size")}' + \
                       f'_LR={kwargs.get("lr")}' + \
                       f'_gamma={kwargs.get("gamma")}' + \
                       f'_tol={kwargs.get("tol")}'
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
    else:
        raise ValueError(
                'Incorrect mode selected; choose "pretrain", "train", or "eval".'
            )

    return savepath_run, serial_run

def load_dataset(fname_dataset, index, send_message=False, transform=None):
    '''
    Arguments:
      fname_dataset: Path to h5 dataset
      index: List of indices to load
      send_message: Boolean
      transform: Data transformation (default: None, )
    '''
    M = len(index)
    with h5py.File(fname_dataset, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/4s/Spectrogram'
        dset = f[DataSpec]
        m, n, o = dset.shape
        m -= 1
        print('--------------------------------------------------------------')
        print(f'H5 file has {m} samples, {n} frequency bins, {o} time bins.')
        print(f'Loading {M} samples...')
        tic = datetime.now()

        np.seterr(all='ignore')
        # X = np.empty([M, n-2, o-173, 1])
        # X = np.zeros([M, 1, 65, 175])
        X = np.zeros([M, 69, 175])
        idx_sample = np.empty([M,], dtype=np.int)
        dset_arr = np.zeros([n, o])
        count = 0
        for i in tqdm(range(M), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            # try:
            if transform == (None or "pixelwise"):
                dset_arr = dset[index[i], :-1, 12:-14] # <---- This by itself doesn't work.
            elif transform == "sample_norm":
                dset_arr /= np.abs(dset_arr).max() # <---- This one works
            elif transform == "sample_norm_cent":
                dset_arr = (dset_arr - dset_arr.mean()) / np.abs(dset_arr).max() # <---- This one works
            elif transform == "sample_std":
                dset_arr = (dset_arr - dset_arr.mean()) / dset_arr.std() # <---- This one throws NaNs for loss in pre-training

            X[count,:,:] = dset_arr
            # X[count,:,:,:] = np.expand_dims(dset_arr, axis=0)
            idx_sample[count,] = int(index[i])
            count += 1

        if transform == "pixelwise":
            X = (X - X.mean(axis=0)) / np.abs(X).max(axis=0)

        X = np.expand_dims(X, axis=1)

        toc = datetime.now()
        msgcontent = f'{M} spectrograms loaded successfully at {toc}.' + \
                     f'\nTime Elapsed = {(toc-tic)}'
        print(msgcontent)
        if send_message:
            msgsubj = 'Data Loaded'
            notify(msgsubj, msgcontent)

    return SeismoDataset(X)

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

def make_pred_configs_batch(loadpath, savepath, overwrite=False):
    def _parse_nclusters(line):
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

    exper = loadpath.split("/")[-1]
    savepath = f"{savepath}/BatchEval_{exper}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    config_ = configparser.ConfigParser()
    config_.read(f"{loadpath}/*.ini")
    print(f"{loadpath}/*.ini")
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
            'n_clusters': _parse_nclusters(run),
            'saved_weights': f'{loadpath}/{run}/{saved_weights}',
            'transform': transform
        }
        with open(fname, 'w') as configfile:
            config.write(configfile)

    print(f'Config Files: {count_wr} written, {count_ow} overwritten, {count_sk} skipped.')

    return savepath

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

    # print('History saved.')

def save_labels(label_list, savepath, serial):
    fname = f'{savepath}/Labels{serial}.csv'
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

    # print('Labels saved.')

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA device available, using GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA device not available, using CPU.')

    return device

# =============================================================================
#  Functions to set/save/get indices of training/validation/prediction samples.
# =============================================================================
def set_TraVal_index(M, fname_dataset, reserve=0.0):
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/4s/Spectrogram'
        m, _, _ = f[DataSpec].shape
        m -= 1
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
    savepath = f'{savepath}TraValIndex_M={M}_Res={reserve}_{serial}.pkl'
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
        DataSpec = '/4s/Spectrogram'
        m, _, _ = f[DataSpec].shape
        m -= 1

    index = np.arange(1, m+1)

    if exclude:
        idx_ex = load_TraVal_index(fname_dataset, indexpath)
        idx_ex = sorted(set(np.concatenate(idx_ex).flatten()))
        index_avail = np.setdiff1d(index, idx_ex)
    else:
        index_avail = index

    # print(len(index_avail))
    # print(M)

    index_tst = np.random.choice(
        index_avail,
        size=int(M * (1+reserve)),
        replace=False
    )
    return index_tst

def set_M(fname_dataset, indexpath, exclude=True):
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/4s/Spectrogram'
        m, _, _ = f[DataSpec].shape
        m -= 1
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
