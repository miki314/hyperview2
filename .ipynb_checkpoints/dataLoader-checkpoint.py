from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %% Dataset paths
DATASET_DIR = Path('./data')

HSI_AIRBORNE_DIR = DATASET_DIR / 'train' / 'hsi_airborne'
HSI_SATELLITE_DIR = DATASET_DIR / 'train' / 'hsi_satellite'
MSI_SATELLITE_TRAIN_DIR = DATASET_DIR / 'train' / 'msi_satellite'
MSI_SATELLITE_TEST_DIR = DATASET_DIR / 'test' / 'msi_satellite'
HSI_SATELLITE_TEST_DIR = DATASET_DIR / 'test' / 'hsi_satellite'

GT_TRAIN_CSV_PATH = DATASET_DIR / 'train_gt.csv'
# %% Load the ground truth measurements
gt_train_df = pd.read_csv(GT_TRAIN_CSV_PATH)
gt_train_df.head()
column_names = ['Fe', 'Zn', 'B', 'Cu', 'S', 'Mn']

def load_msi_data(directory):
    files = sorted(directory.glob('*.npz'))
    msi_data = []
    for file in enumerate(files):
        file_name = file[1]
        with np.load(file_name) as npz:
            arr = np.ma.MaskedArray(**npz)
            mean_pixel = np.mean(arr, axis=(1, 2))  # mean over all pixels
            msi_data.append(mean_pixel)

    msi_data = np.array(msi_data)
    return msi_data

def get_Xy_train():
    with open("wavelengths.json", "r") as file:
        wavelengths = json.load(file)
        aerial = list(wavelengths['hsi_aerial_wavelengths'].values())
        hsi_satellite = list(wavelengths['hsi_satellite_wavelengths'].values())
        msi_satellite = list(wavelengths['msi_satellite_wavelengths'].values())

    i = 0

    hsi_satellite_index = []
    for j, wavelength in enumerate(hsi_satellite):
        if wavelength >= msi_satellite[i]:
            if i==0 or abs(hsi_satellite[j]-msi_satellite[i]) < abs(hsi_satellite[j-1]-msi_satellite[i]):
                hsi_satellite_index.append(j)
            else:
                hsi_satellite_index.append(j-1)
            
            i += 1
            if i == len(msi_satellite): break

    i = 0

    aerial_index = []
    for j, wavelength in enumerate(aerial):
        if wavelength >= msi_satellite[i]:
            if i==0 or abs(aerial[j]-msi_satellite[i]) < abs(aerial[j-1]-msi_satellite[i]):
                aerial_index.append(j)
            else:
                aerial_index.append(j-1)
            
            i += 1
            if i == len(msi_satellite): break

    gt_train_df = pd.read_csv(GT_TRAIN_CSV_PATH)
    X_msi_train = load_msi_data(MSI_SATELLITE_TRAIN_DIR)
    X_msi_test = load_msi_data(MSI_SATELLITE_TEST_DIR)

    X_airborne = load_msi_data(HSI_AIRBORNE_DIR)
    X_airborne = X_airborne[:,aerial_index]

    X_hsi = load_msi_data(HSI_SATELLITE_DIR)
    X_hsi = X_hsi[:,hsi_satellite_index]
    y= gt_train_df[column_names].values

    X= np.vstack((X_msi_train, X_hsi))
    X= np.vstack((X, X_airborne))

    y_tmp = np.vstack((y, y))
    y= np.vstack((y, y_tmp))

    return (X,y)