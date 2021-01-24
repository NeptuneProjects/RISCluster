#!/usr/bin/env python3

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

from dotenv import load_dotenv
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
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


class LabelCatalogue(object):
    def __init__(self, paths, label_list=None):
        self.paths = paths
        self.freq = None
        self.df = self.build_df(self.paths)
        if label_list is not None:
            self.label_list = label_list.sort()
        else:
            self.label_list = np.sort(pd.unique(self.df["label"]))
        self.station_list = pd.unique(self.df["station"])


    def build_df(self, paths):
        data1 = pd.read_csv(paths[0]).drop(columns=["Index"])
        data2 = pd.read_csv(self.paths[1]).drop(columns=["idx"])
        df = pd.concat(
            [data1, data2],
            axis=1
        ).drop(
            columns=[
                "channel",
                "dt_on",
                "dt_off",
                "fs",
                "delta",
                "npts",
                "STA",
                "LTA",
                "on",
                "off",
                "spec_start",
                "spec_stop"
            ]
        ).rename(columns={"dt_peak": "time"})
        df["time"] = df["time"].astype("datetime64")
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df


    def gather_counts(self, station, freq="month", label_list=None):
        if freq == "month":
            freqcode = "1M"
        elif freq == "day":
            freqcode = "1D"
        elif freq == "hour":
            freqcode = "1H"
        self.freq = freq
        if (label_list is not None) and (max(label_list) > max(self.label_list)):
            raise ValueError("label_list includes impossibly high label.")
        else:
            label_list = self.label_list
        for i, label in enumerate(label_list):
            mask = (self.df["station"] == station) & (self.df['label'] == label)
            subset = self.df.loc[mask].drop(columns=["network","station","peak","unit"])
            counts = subset.resample(freqcode).count().rename(columns={"label": f"{label+1}"})
            if i == 0:
                df = counts
            else:
                df = pd.concat([df, counts], axis=1)
        return df.fillna(0).astype("int").sort_index()


    def seasonal_statistics(self, mode=None):
        if mode is not None:
            count_summer15 = np.empty((len(self.label_list),))
            count_winter15 = np.empty((len(self.label_list),))
            count_summer16 = np.empty((len(self.label_list),))
            count_winter16 = np.empty((len(self.label_list),))
            total = np.empty((len(self.label_list),))

            for j, label in enumerate(self.label_list):
                mask_label = self.df["label"] == label
                subset = self.df.loc[mask_label]
                total_count = len(subset.index)
                mask_summer15 = (subset.index >= datetime(2015,1,1)) & (subset.index < datetime(2015,4,1))
                mask_winter15 = (subset.index >= datetime(2015,6,1)) & (subset.index < datetime(2015,9,1))
                mask_summer16 = (subset.index >= datetime(2016,1,1)) & (subset.index < datetime(2016,4,1))
                mask_winter16 = (subset.index >= datetime(2016,6,1)) & (subset.index < datetime(2016,9,1))

                count_summer15[j] = 100 * len(subset.loc[mask_summer15].index) / total_count
                count_winter15[j] = 100 * len(subset.loc[mask_winter15].index) / total_count
                count_summer16[j] = 100 * len(subset.loc[mask_summer16].index) / total_count
                count_winter16[j] = 100 * len(subset.loc[mask_winter16].index) / total_count
                total[j] = total_count

            return pd.DataFrame({"total": total, "JFMTotal": count_summer15 + count_summer16, "JFM15": count_summer15, "JFM16": count_summer16, "JJATotal": count_winter15 + count_winter16, "JJA15": count_winter15, "JJA16": count_winter16})
        else:
            count_summer = np.empty((len(self.label_list),))
            count_winter = np.empty((len(self.label_list),))
            for j, label in enumerate(self.label_list):
                mask_label = self.df["label"] == label
                subset = self.df.loc[mask_label]
                total_count = len(subset.index)
                mask_summer = ((subset.index >= datetime(2015,1,1)) & (subset.index < datetime(2015,4,1))) | \
                    ((subset.index >= datetime(2016,1,1)) & (subset.index < datetime(2016,4,1)))
                mask_winter = ((subset.index >= datetime(2015,6,1)) & (subset.index < datetime(2015,9,1))) | \
                    ((subset.index >= datetime(2016,6,1)) & (subset.index < datetime(2016,9,1)))

                count_winter[j] = 100 * len(subset.loc[mask_winter].index) / total_count
                count_summer[j] = 100 * len(subset.loc[mask_summer].index) / total_count
            return pd.DataFrame({"JFM": count_summer, "JJA": count_winter})


    def station_statistics(self):
        count = np.empty((len(self.station_list),))
        percent = np.empty((len(self.station_list),))
        total_count = len(self.df.index)
        label_matrix = np.empty((len(self.station_list),len(self.label_list)))
        for i, station in enumerate(self.station_list):
            mask_station = self.df["station"] == station
            subset_station = self.df.loc[mask_station]
            count[i] = len(subset_station.index)
            percent[i] = 100 * len(subset_station.index) / total_count
            for j, label in enumerate(self.label_list):
                mask_label = subset_station["label"] == label
                subset_label = subset_station.loc[mask_label]
                label_matrix[i, j] = len(subset_label.index)

        label_matrix_dict = []

        df = pd.DataFrame({"station": self.station_list, "N": count, "percent": percent})
        df = pd.concat([df, pd.DataFrame(label_matrix)], axis=1)
        for col in [1, range(3, 3 + len(self.label_list))]:
            df.iloc[:, col] = df.iloc[:, col].astype("int")
        return df.sort_values(by="station", ignore_index=True)


    def amplitude_statistics(self):
        columns = ["Class", "Mean", "Median", "Standard Deviation", "Maximum"]
        stats = []
        for label in self.label_list:
            subset = self.df["peak"].loc[self.df["label"] == label].abs()
            stats.append((label+1, subset.mean(), subset.median(), subset.std(), subset.max()))

        amp_stats = pd.DataFrame(stats, columns=columns).sort_values(by=["Class"], ignore_index=True)
        return amp_stats.set_index("Class")


    # def get_peak_frequency(
    #         path_to_labels,
    #         path_to_catalogue,
    #         fname_dataset,
    #         batch_size=2048,
    #         workers=12
    #     ):
    #     labels = pd.read_csv(path_to_labels, index_col=0)
    #     label_list = pd.unique(labels["label"])
    #     _, _, fvec = load_images(fname_dataset, [[0]])
    #
    #     class_avg_maxfreq = np.zeros(len(label_list))
    #     for j, label in enumerate(label_list):
    #         mask = labels.label == label
    #         labels_subset = labels.loc[mask]
    #         dataset = H5SeismicDataset(
    #             fname_dataset,
    #             transform = transforms.Compose(
    #                 [SpecgramShaper(), SpecgramToTensor()]
    #             )
    #         )
    #         subset = Subset(dataset, labels_subset.index)
    #         dataloader = DataLoader(subset, batch_size=batch_size, num_workers=workers)
    #         batch_avg_maxfreq = np.zeros(len(dataloader))
    #         pbar = tqdm(
    #             dataloader,
    #             bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
    #             desc=f"Label {j+1}/{len(label_list)} (Class {label})",
    #             leave=True,
    #             position=0
    #         )
    #         for i, batch in enumerate(pbar):
    #             _, X = batch
    #             maxfreqind = torch.max(torch.sum(X, 3) / X.size(3), 2).indices
    #             maxfreqind = maxfreqind.detach().cpu().numpy()
    #             maxfreq = fvec[maxfreqind]
    #             batch_avg_maxfreq[i] = maxfreq.mean()
    #
    #         class_avg_maxfreq[j] = batch_avg_maxfreq.sum() / len(dataloader)
    #         print(f"Avg. Peak Frequency: {class_avg_maxfreq[j]:.2f} Hz", flush=True)
    #
    #     peak_freqs = pd.DataFrame({"Class": label_list, "Avg_Peak_Freq": class_avg_maxfreq}).sort_values(by=["Class"], ignore_index=True)
    #     return peak_freqs.set_index("Class")


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
        if self.transform is not None:
            if self.transform == "sample_norm":
                X /= np.abs(X).max(axis=(0,1))
            elif self.transform == "sample_norm_cent":
                # X = (X - X.mean(axis=(0,1))) / \
                # np.abs(X).max(axis=(0,1))
                X = (X - X.mean()) / np.abs(X).max()
            else:
                raise ValueError("Unsupported transform.")
        else:
            print("Test failed.")

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


def get_amplitude_statistics(path_to_labels, path_to_catalogue):
    labels = pd.read_csv(path_to_labels, index_col=0)
    amp_pk = pd.read_csv(path_to_catalogue, usecols=["peak"])

    data = pd.concat([labels, amp_pk], axis=1)
    label_list = np.sort(pd.unique(labels["label"]))

    columns = ["Class", "Mean", "Median", "Standard Deviation", "Maximum"]
    stats = []
    for label in label_list:
        subset = data["peak"].loc[data["label"] == label].abs()
        stats.append((label+1, subset.mean(), subset.median(), subset.std(), subset.max()))

    amp_stats = pd.DataFrame(stats, columns=columns).sort_values(by=["Class"], ignore_index=True)
    return amp_stats.set_index("Class")


def get_peak_frequency(
        path_to_labels,
        path_to_catalogue,
        fname_dataset,
        batch_size=2048,
        workers=12
    ):
    labels = pd.read_csv(path_to_labels, index_col=0)
    label_list = pd.unique(labels["label"])
    _, _, fvec = load_images(fname_dataset, [[0]])

    class_avg_maxfreq = np.zeros(len(label_list))
    for j, label in enumerate(label_list):
        mask = labels.label == label
        labels_subset = labels.loc[mask]
        dataset = H5SeismicDataset(
            fname_dataset,
            transform = transforms.Compose(
                [SpecgramShaper(), SpecgramToTensor()]
            )
        )
        subset = Subset(dataset, labels_subset.index)
        dataloader = DataLoader(subset, batch_size=batch_size, num_workers=workers)
        batch_avg_maxfreq = np.zeros(len(dataloader))
        pbar = tqdm(
            dataloader,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
            desc=f"Label {j+1}/{len(label_list)} (Class {label})",
            leave=True,
            position=0
        )
        for i, batch in enumerate(pbar):
            _, X = batch
            maxfreqind = torch.max(torch.sum(X, 3) / X.size(3), 2).indices
            maxfreqind = maxfreqind.detach().cpu().numpy()
            maxfreq = fvec[maxfreqind]
            batch_avg_maxfreq[i] = maxfreq.mean()

        class_avg_maxfreq[j] = batch_avg_maxfreq.sum() / len(dataloader)
        print(f"Avg. Peak Frequency: {class_avg_maxfreq[j]:.2f} Hz", flush=True)

    peak_freqs = pd.DataFrame({"Class": label_list, "Avg_Peak_Freq": class_avg_maxfreq}).sort_values(by=["Class"], ignore_index=True)
    return peak_freqs.set_index("Class")


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
        fvec = dset[0, 0:87, 0]
        tvec = dset[0, 87, 1:]
    X = X[:, :-1, 1:]

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
