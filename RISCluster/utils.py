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
from twilio.rest import Client

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
        savepath_run = []
        for label in range(n_clusters):
            savepath_cluster = f'{savepath}Cluster{label:02d}'
            if not os.path.exists(savepath_cluster):
                os.makedirs(savepath_cluster)
            savepath_run.append(savepath_cluster)
    else:
        raise ValueError('Incorrect mode selected; choose "pretrain", "train", or "eval".')

    return savepath_run, serial_run

def init_exp_env(mode, savepath):
    serial_exp = datetime.now().strftime('%Y%m%dT%H%M%S')
    if mode == 'pretrain':
        savepath_exp = savepath + 'Models/AEC/Exp' + serial_exp + '/'
    elif mode == 'train':
        savepath_exp = savepath + 'Models/DCEC/Exp' + serial_exp + '/'
    elif mode == 'predict':
        savepath_exp = savepath + 'Trials/Exp' + serial_exp + '/'
    else:
        raise ValueError('Incorrect mode selected; choose "pretrain", "train", or "eval".')
    if not os.path.exists(savepath_exp):
        os.makedirs(savepath_exp)
    print('New experiment file structure created at:\n'
          f'{savepath_exp}')
    return savepath_exp, serial_exp

def load_weights(model, fname, device):
    model.load_state_dict(torch.load(fname, map_location=device))
    model.eval()
    print(f'Weights loaded to {device}')
    return model

def load_data(fname_dataset, M, index, send_message=True):
    '''
    *M* random spectrograms are read in and pre-processed iteratively as
    follows:
    1. Open the file and specified h5 dataset.
    2. Read the *m*th spectrogram from file.  Each spectrogram is of shape
    (*n*, *o*).
    3. Omit unnecessary data when assigning data into memory to cut down on
    memory usage and to make subsequent convolutions and transpositions
    straightforward with regard to data dimensions.  Specifically, index *n*[0]
    contains low frequency data that is of no interest for our high-frequency
    analaysis; *n*[65] is a vector of time values; and *o*[0] is a vector of
    frequency values.  Additional *o* indices may be omitted to reduce the
    length of spectrogram being analyzed.
    4. Add a fourth axis for the amplitude dimension.  This is requried for
    subsequent steps.
    At the conclusion of reading in and pre-processing, data has been omitted
    in such a way that the final shape of the array is
    (*m*, *n*, *o*, *p*) = (*M*, 64, 128, 1).
    '''
    with h5py.File(fname_dataset, 'r') as f:
        #samples, frequency bins, time bins, amplitude
        DataSpec = '/30sec/Spectrogram'
        dset = f[DataSpec]
        m, n, o = dset.shape
        print('--------------------------------------------------------------')
        print(f'H5 file has {m} samples, {n} frequency bins, {o} time bins.')
        print(f'Loading {M} samples...')
        tic = datetime.now()

        np.seterr(all='raise')
        # X = np.empty([M, n-2, o-173, 1])
        X = torch.empty([M, 1, n-2, o-173])
        idx_sample = np.empty([M,], dtype=np.int)
        dset_arr = np.empty([n, o])
        # dset_arr = torch.empty([1, n, o])
        count = 0
        for i in range(M):
            try:
                dset_arr = dset[index[i], 1:-1, 1:129]
                dset_arr /= dset_arr.max()
                X[count,:,:,:] = torch.from_numpy(np.expand_dims(dset_arr,
                                                                 axis=0))
                idx_sample[count,] = int(index[i])
                count += 1
            except:
                print('Numpy "Divide-by-zero Warning" raised, '
                      'skipping spectrogram.')
                print('Sample Index = {}'.format(index[i]))
                print(dset[index[i], 1:-1, 1:129])
                pass

            print('    %.2f' % (float(100*i/(M-1))) + '% complete.', end='\r')
        toc = datetime.now()
        print(f'\nTime elapsed = {toc-tic}')

    m, p, n, o = list(X.size())
    print(f'Shape of output is {(m, p, n, o)}')
    if send_message:
        msgsubj = 'Data Loaded'
        msgcontent = f'''{M} spectrograms loaded successfully at {toc}.
Time Elapsed = {(toc-tic)}'''
        notify(msgsubj, msgcontent)
    return X, m, p, n, o, idx_sample

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
        with smtplib.SMTP_SSL('smtp.gmail.com',
                              port=465, context=context) as s:
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
        msgcontent = '*' + msgsubj + '*\n' + msgcontent
        client.messages.create(body=msgcontent,
                               from_=orig_whatsapp_number,
                               to=rx_whatsapp_number)
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
        fname = savepath + f'AEC_History{run_serial}.csv'
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
        fname = savepath + f'DCEC_History{run_serial}.csv'
        d2 = training_history

    with open(fname, 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(d2.keys())
        w.writerows(zip(*d2.values()))
    print('History saved.')

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA device available, using GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA device not available, using CPU.')
    return device

def set_loading_index(M, fname_dataset, reserve=0.02):
    with h5py.File(fname_dataset, 'r') as f:
        DataSpec = '/30sec/Spectrogram'
        m, _, _ = f[DataSpec].shape
    index = np.random.choice(np.arange(1,m), size=int(M * (2 + reserve)), replace=False)
    split = int(len(index)/2)
    index_test = index[split:]
    index_train_val = index[0:split]
    split_pct = 0.8
    split = int(split_pct * len(index_train_val))
    index_train = index_train_val[0:split]
    index_val = index_train_val[split:]
    return index_train, index_val, index_test
