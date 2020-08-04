import csv
from datetime import datetime
from email.message import EmailMessage
import os
import pickle
import smtplib
import ssl

from dotenv import load_dotenv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from twilio.rest import Client

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

def calc_tuning_runs(hyperparameters):
    tuning_runs = 1
    for key in hyperparameters:
        tuning_runs *= len(hyperparameters[key])

    return(tuning_runs)

def init_output_env(savepath, mode, **kwargs):
    serial_run = datetime.now().strftime('%Y%m%dT%H%M%S')
    if mode == 'pretrain':
        savepath_run = f'{savepath}Run' + \
                       f'_BatchSz={kwargs.get("batch_size")}' + \
                       f'_LR={kwargs.get("lr")}/'
        if not os.path.exists(savepath_run):
            os.makedirs(savepath_run)
    elif mode == 'train':
        savepath_run = f'{savepath}Run' + \
                       f'_BatchSz={kwargs.get("batch_size")}' + \
                       f'_LR={kwargs.get("lr")}' + \
                       f'_gamma={kwargs.get("gamma")}' + \
                       f'_tol={kwargs.get("tol")}/'
    elif mode == 'predict':
        n_clusters = kwargs.get('n_clusters')
        with open(f'{savepath}{n_clusters}_Clusters', 'w') as f:
            pass
        savepath_run = []
        for label in range(n_clusters):
            savepath_cluster = f'{savepath}Cluster{label:02d}/'
            if not os.path.exists(savepath_cluster):
                os.makedirs(savepath_cluster)
            savepath_run.append(savepath_cluster)
    else:
        raise ValueError(
                'Incorrect mode selected; choose "pretrain", "train", or "eval".'
            )

    return savepath_run, serial_run

def init_exp_env(mode, savepath):
    serial_exp = datetime.now().strftime('%Y%m%dT%H%M%S')
    if mode == 'pretrain':
        savepath_exp = f'{savepath}Models/AEC/Exp{serial_exp}/'
    elif mode == 'train':
        savepath_exp = f'{savepath}Models/DCEC/Exp{serial_exp}/'
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

def load_dataset(fname_dataset, index, send_message=False):
    M = len(index)
    with h5py.File(fname_dataset, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/30sec/Spectrogram'
        dset = f[DataSpec]
        m, n, o = dset.shape
        m -= 1
        print('--------------------------------------------------------------')
        print(f'H5 file has {m} samples, {n} frequency bins, {o} time bins.')
        print(f'Loading {M} samples...')
        tic = datetime.now()

        np.seterr(all='raise')
        # X = np.empty([M, n-2, o-173, 1])
        X = np.empty([M, 1, n-2, o-173])
        idx_sample = np.empty([M,], dtype=np.int)
        dset_arr = np.empty([n, o])
        count = 0
        for i in tqdm(range(M)):
            try:
                dset_arr = dset[index[i], 1:-1, 1:129]
                dset_arr /= dset_arr.max()
                X[count,:,:,:] = np.expand_dims(dset_arr,axis=0)
                idx_sample[count,] = int(index[i])
                count += 1
            except:
                print('Numpy "Divide-by-zero Warning" raised, '
                      'skipping spectrogram.')
                print(f'Sample Index = {index[i]}')
                print(dset[index[i], 1:-1, 1:129])
                pass

        toc = datetime.now()
        print(f'\nTime elapsed = {toc-tic}')

    return SeismoDataset(X)

def load_weights(model, fname, device):
    model.load_state_dict(torch.load(fname, map_location=device))
    model.eval()
    print(f'Weights loaded to {device}')

    return model

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

def save_exp_config(savepath, serial, parameters, hyperparameters):
    fname = f'{savepath}ExpConfig{serial}'
    if hyperparameters is not None:
        configs = [parameters, hyperparameters]
    else:
        configs = parameters
    with open(f'{fname}.txt', 'w') as f:
        f.write(str(configs))
    with open(f'{fname}.pkl', 'wb') as f:
        pickle.dump(configs, f)

def save_history(training_history, validation_history, savepath, run_serial):
    if validation_history is not None:
        fname = f'{savepath}AEC_History{run_serial}.csv'
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
        fname = f'{savepath}DCEC_History{run_serial}.csv'
        d2 = training_history

    with open(fname, 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(d2.keys())
        w.writerows(zip(*d2.values()))

    print('History saved.')

def save_labels(label_list, savepath, serial):
    fname = f'{savepath}Labels{serial}.csv'
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

    print('Labels saved.')

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
        DataSpec = '/30sec/Spectrogram'
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
        DataSpec = '/30sec/Spectrogram'
        m, _, _ = f[DataSpec].shape
        m -= 1

    index = np.arange(1, m)

    if exclude:
        idx_ex = load_TraVal_index(fname_dataset, indexpath)
        idx_ex = sorted(set(np.concatenate(idx_ex).flatten()))
        index_avail = [i for i in list(index) if i not in idx_ex]
    else:
        index_avail = index

    index_val = np.random.choice(
        index_avail,
        size=int(M * (1+reserve)),
        replace=False
    )
    return index_val

def set_M(fname_dataset, indexpath, exclude=True):
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/30sec/Spectrogram'
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
