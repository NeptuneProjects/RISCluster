#!/usr/bin/env python3

import matplotlib
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat

matplotlib.rc('font', **{'size': 18})


def read_image(file, n_header, n_row, n_col):
    image = np.fromfile(file, dtype='uint8')
    return np.reshape(image[n_header:], (n_row, n_col))


def get_representatives(image, min_row, max_row, min_col, max_col):
    image_copy = image.copy().astype(float)
    image_copy[image_copy > 250] = np.nan
    return np.nanmean(image[min_row:max_row, min_col:max_col]) / 250


def plot_representatives(dates, representatives):
    _, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, representatives)
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid('on')
    plt.show()


def plot_map(image, min_col, max_col, min_row, max_row):
    image_clip = image.copy()
    image_clip[image_clip > 250] = 250

    cm = plt.get_cmap('Blues_r')
    image_color = cm(image_clip)
    image_color = np.expand_dims(image_color, axis=3)
    new_image = np.ones((image_color.shape[0], image_color.shape[1], 4)) * 1
    for i in range(len(image_color)):
        for j in range(len(image_color[0])):
            for k in range(3):
                new_image[i, j, k] = image_color[i, j, k]
    for i in range(len(image_color)):
        for j in range(len(image_color[0])):
            if image[i, j] == 253:
                new_image[i, j, :] = np.array([0, 0, 0, 1])
            elif image[i, j] > 250:
                new_image[i, j, :] = np.array([0.5, 0.5, 0.5, 1])
    for i in [min_row - 1, min_row, min_row + 1,
              max_row - 2, max_row - 1, max_row]:
        for j in range(min_col, max_col):
            new_image[i, j, :] = np.array([1, 0, 0, 1])
    for j in [min_col - 1, min_col, min_col + 1,
              max_col - 2, max_col - 1, max_col]:
        for i in range(min_row, max_row):
            new_image[i, j, :] = np.array([1, 0, 0, 1])

    _, ax = plt.subplots(figsize=(4, 4))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    _ = ax.imshow(new_image, cmap='Blues_r')
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    plt.show()


def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(figsize=(7.72, 0.9))
    plt.subplots_adjust(left=0.05, bottom=0.5, right=0.95, top=1)
    ax.set_yticks([])
    ax.imshow([colors], extent=[0, 1, 0, 0.02])
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(val) for val in vals])
    ax.set_xlabel('Sea-ice concentration')
    plt.show()


def main():
    start_date = pd.Timestamp(2014, 12, 1)
    end_date = pd.Timestamp(2016, 12, 1)
    n_header = 300  # The leading 300 bytes are headers
    n_row, n_col = 332, 316  # Number of rows and columns of the image
    min_col, max_col = 120, 180  # Columns of interest
    min_row, max_row = 220, 280  # Rows of interest
    file_dir = ('/Users/williamjenkins/Research/Data/NSIDC-0051')

    # Calculate representative ice concentration of the selected region
    dates = pd.date_range(start=start_date, end=end_date).to_pydatetime()
    representatives = np.zeros((len(dates),))
    for i, date in enumerate(dates):
        file = (f"{file_dir}/nt_{date.year}{date.month:02d}{date.day:02d}_f17_v1.1_s.bin")
        image = read_image(file, n_header, n_row, n_col)
        representatives[i] = (get_representatives(image, min_row, max_row, min_col, max_col))

    mdic = {
        "C": representatives,
        "date": [dates[i].strftime('%Y-%b-%d %H:%M:%S') for i in range(len(dates))]
    }
    savemat(f"{file_dir}/NSIDC-0051.mat", mdic)


if __name__ == '__main__':
    main()
