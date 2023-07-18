from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset


class SeqDataset(Dataset):
    """ Sequence dataset to (input, target) dataset. """

    def __init__(self, dataset: Dataset, input_len: int, target_len: int, step='input_len'):
        self.dataset = dataset
        self.input_len = input_len
        self.target_len = target_len
        self.step = getattr(self, step) if isinstance(step, str) else step

    @property
    def seq_len(self): return self.input_len + self.target_len

    def __getitem__(self, item):
        if isinstance(item, slice):
            return (self[i] for i in range(*item.indices(len(self))))

        start = item * self.step
        split = start + self.input_len  # split point between input and target
        stop = split + self.target_len
        X = self.dataset[start:split]
        Y = self.dataset[split:stop]
        return X, Y

    def __len__(self):
        return (len(self.dataset) - self.seq_len) // self.step + 1


class FinanceDataset(Dataset):
    def __init__(self, path, device=None):
        """
		A class to convert a pandas dataframe into a pytorch compatible dataset.
		:param path: The path to csv file
		:param device: the torch.device (cpu or cuda) to load the data onto
		"""
        df = csv2pandas(path)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        columns = df.columns.values  # include all columns
        tensor = torch.as_tensor(df[columns].values, dtype=torch.float32, device=device)
        self.len = len(tensor)
        self.dic = dict(zip(columns, tensor.T))

    def __getitem__(self, item):
        return {
            key: value[item]
            for key, value in self.dic.items()
        }

    def __len__(self):
        return self.len


def csv2pandas(path):
    """Converts the csv file into pandas dataframe with relevant features."""
    df = pd.read_csv(path, index_col=0, names=['datetime', 'ts1', 'ts2'], header=None)
    # df = df.filter(regex=series)
    df.index = pd.to_datetime(df.index - 719529, unit='d')  # converting from Matlab datetime format
    for ts in ['ts1', 'ts2']:
        df[ts] = np.log(df[ts])

    '''Cleaning Data'''
    # Why not interpolate?
    # Might create garbage data
    # Why not drop nan?
    # Do not drop good data

    # Get individual time series features
    df = pd.merge(get_time_series_features(df, 'ts1'),
                  get_time_series_features(df, 'ts2'),
                  left_index=True,
                  right_index=True)

    # Get global features
    add_global_features(df)

    return df


def get_time_series_features(df, series='ts1'):
    """Consolidate information per time bucket by
    resampling, cleaning NaNs and adding features"""
    # Done at a daily scale, can be done at other frequencies.
    # You may want to adjust add_global_features() as per this frequency.
    dfd = df.resample('D')
    df = dfd[series].agg(**{
        f'volume_{series}': 'count',
        f'mean_{series}': 'mean',
        f'var_{series}': 'var',
    })
    # Filling Nans
    df[f'mean_{series}'] = df[f'mean_{series}'].ffill()
    # df['mean_norm'] = df['mean']/np.abs(np.max(df['mean']))
    df[f'var_{series}'] = df[f'var_{series}'].fillna(0.0)

    # Addding time series features
    df[f'mean_diff_{series}'] = df[f'mean_{series}'].diff()
    df.drop(df.index[0], inplace=True)

    df[f'mean_binary_{series}'] = \
        (np.multiply([df[f'mean_diff_{series}'] >= 0], 1) + np.multiply([df[f'mean_diff_{series}'] < 0], -1))[0]
    df[f'volume_{series}'] = df[f'volume_{series}'] / df[f'volume_{series}'].max()  # normalise volume
    return df


def add_global_features(df):
    """This function generates global features"""
    df[f'day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df[f'day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df[f'day_of_month_sin'] = np.sin(2 * np.pi * df.index.day / df.index.daysinmonth)
    df[f'day_of_month_cos'] = np.cos(2 * np.pi * df.index.day / df.index.daysinmonth)
    df[f'day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / (365 + df.index.is_leap_year))
    df[f'day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / (365 + df.index.is_leap_year))

