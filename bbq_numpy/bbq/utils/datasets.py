import pandas as pd
import os
import numpy as np

import matplotlib.image as mpimg
from .image import rgb2gray

DEFAULT_DIR = os.path.join("bbq", "datasets")


def mauna_loa(root_dir=DEFAULT_DIR, raw_data=False,
              **read_csv_kwargs):
    """
    Loads the Mauna Loa C02 dataset from 1965 to 2016
    (years with complete data...)
    :param raw_data:
    :param root_dir:
    :param read_csv_kwargs:
    :return:
    """
    if raw_data:
        file_path = os.path.join(root_dir, "raw_datasets",
                                 "mauna-loa-c02-1965-2016.csv")
        dataset = pd.read_csv(file_path, usecols=[2, 3], skiprows=54,
                              engine="python", **read_csv_kwargs)
        data = np.array(dataset.values)
        return data
    else:
        file_path = os.path.join(root_dir, "co2")
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test


def airline_passengers(root_dir=DEFAULT_DIR, raw_data=False,
                       **read_csv_kwargs):
    """
    Loads the "international-airline-passenger" dataset from
    https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line
    :param raw_data:
    :param root_dir:
    :param read_csv_kwargs:
    :return:
    """
    if raw_data:
        file_path = os.path.join(root_dir, "raw_datasets",
                                 "international-airline-passengers.csv")
        dataset = pd.read_csv(file_path, usecols=[1], engine='python',
                              skipfooter=3,
                              **read_csv_kwargs)
        data_raw = np.hstack([np.array(dataset.index).reshape(-1, 1),
                              np.array(dataset.values).reshape(-1, 1)])
        return data_raw
    else:
        file_path = os.path.join(root_dir, "airline_passengers")
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test


def concrete(root_dir=DEFAULT_DIR, raw_data=False,
             **read_csv_kwargs):
    """
    Loads the "Concrete Compressive Strength" dataset from
    https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
    :param raw_data:
    :param root_dir:
    :param read_csv_kwargs:
    :return:
    """
    if raw_data:
        file_path = os.path.join(root_dir, "raw_datasets",
                                 "Concrete_Data-1.csv")
        dataset = pd.read_csv(file_path, engine='python',
                              **read_csv_kwargs)
        return np.array(dataset.values)
    else:
        file_path = os.path.join(root_dir, "concrete")
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test


def airfoil_noise(root_dir=DEFAULT_DIR, raw_data=False,
                  **read_csv_kwargs):
    """
    Loads the "Airfoil self-noise" dataset from
    https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
    :param raw_data:
    :param root_dir:
    :param read_csv_kwargs:
    :return:
    """
    if raw_data:
        file_path = os.path.join(root_dir, "raw_datasets",
                                 "airfoil_self_noise.csv")
        dataset = pd.read_csv(file_path, engine='python',
                              **read_csv_kwargs)
        return np.array(dataset.values)
    else:
        file_path = os.path.join(root_dir, "airfoil_noise")
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test


def textures_2D(root_dir=DEFAULT_DIR, texture_name="pores", raw_data=False):
    """
    TOTAL = 1690
    TRAIN: 12675
    TEST: 4225

    res: (130 across, 130 up)
    i.e. (130,130)

    we have a cutout of (65,65)

    #1    (0 to 129 , 130)  then  (0 to 31, 32)
    #2    (0 to 31, 32)     then  (32 to 96, 65)
    #3    (97 to 129,  33)  then  (32 to 96,  65)
    #4    (0 to 129,  130)   then  (97 to 129, 33)
    """
    if raw_data:
        rgb_img = mpimg.imread(os.path.join(root_dir, "raw_datasets",
                                            '{}.png'.format(texture_name)))
        g_img = rgb2gray(rgb_img)

        # The training set
        x_trn_1 = np.mgrid[
                0:129:complex(0, 130),
                0:31:complex(0, 32)].reshape(2, -1).T.astype(np.int)
        x_trn_2 = np.mgrid[
                0:31:complex(0, 32),
                32:96:complex(0, 65)].reshape(2, -1).T.astype(np.int)
        x_trn_3 = np.mgrid[
                97:129:complex(0, 33),
                32:96:complex(0, 65)].reshape(2, -1).T.astype(np.int)
        x_trn_4 = np.mgrid[
                0:129:complex(0, 130),
                97:129:complex(0, 33)].reshape(2, -1).T.astype(np.int)

        x_trn = np.vstack((x_trn_1, x_trn_2, x_trn_3, x_trn_4))

        x_tst = np.mgrid[
                32:96:complex(0, 65),
                32:96:complex(0, 65)].reshape(2, -1).T.astype(np.int)

        y_trn = g_img[x_trn[:, 0], x_trn[:, 1]].reshape(-1, 1)
        y_tst = g_img[x_tst[:, 0], x_tst[:, 1]].reshape(-1, 1)

        return x_trn, x_tst, y_trn, y_tst
    else:
        file_path = os.path.join(root_dir, "textures_2D", texture_name)
        train = np.genfromtxt(os.path.join(file_path, "train.csv"),
                              delimiter=",")
        test = np.genfromtxt(os.path.join(file_path, "test.csv"),
                             delimiter=",")
        return train, test
