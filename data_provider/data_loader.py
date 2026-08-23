

import os
import random
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features
from .monash_data_utils import convert_tsf_to_dataframe

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = "h"

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):
        
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        from statsmodels.tsa.stattools import adfuller
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y))

        self.data_stamp = data_stamp

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))

        return seq_x, seq_y, seq_x_mark, seq_y_mark, idx, timestamps1, timestamps2, len(self.data_x)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = "t"

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        from statsmodels.tsa.stattools import adfuller
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y))

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))

        return seq_x, seq_y, seq_x_mark, seq_y_mark, idx, timestamps1, timestamps2, len(self.data_x)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        from statsmodels.tsa.stattools import adfuller
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y))

        self.data_stamp = data_stamp

    def __getitem__(self, index):

        # ========================================================================
        # if (self.set_type==0) and (self.args.dataset_name in ["ECL", "traffic"]) and (self.args.model in ["DDPM"]):
        #     index = np.random.randint(len(self.data_x) - self.seq_len - self.pred_len + 1)

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))

        return seq_x, seq_y, seq_x_mark, seq_y_mark, idx, timestamps1, timestamps2, len(self.data_x)
        
    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
        

class Dataset_Custom_S(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.num_vars = np.shape(self.data_x)[1]
        self.ts_len = np.shape(self.data_x)[0]

        self.seq_i = {}
        self.seq_j = {}
        self.global_inx = 0
        for var_i in range(self.num_vars):
            for t_i in range(self.ts_len-self.args.seq_len-self.args.pred_len):

                self.seq_i[self.global_inx] = var_i  
                self.seq_j[self.global_inx] = t_i
                self.global_inx += 1

        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y),self.global_inx)

        self.data_stamp = data_stamp


    def __getitem__(self, index):

        var_i = self.seq_i[index]
        index = self.seq_j[index]

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = np.expand_dims(self.data_x[s_begin:s_end, var_i], axis=1)
        seq_y = np.expand_dims(self.data_y[r_begin:r_end, var_i], axis=1)
        seq_x_mark = np.expand_dims(self.data_stamp[s_begin:s_end, var_i], axis=1)
        seq_y_mark = np.expand_dims(self.data_stamp[r_begin:r_end, var_i], axis=1)

        # print(self.global_inx, var_i, index, np.shape(seq_x), np.shape(seq_y))
        # ================================================================================
        # global time steps
        # idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))

        return seq_x, seq_y, seq_x_mark, seq_y_mark, var_i, timestamps1, timestamps2, len(self.data_x)
        
    def __len__(self):

        if (self.set_type==0): 
            return min(5000, self.global_inx)
        else:
            return self.global_inx 


class Dataset_wind(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = args.target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y))

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))

        return seq_x, seq_y, seq_x_mark, seq_y_mark, idx, timestamps1, timestamps2, len(self.data_x)
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Monash(Dataset):
    def __init__(self, args, root_path, flag='train', features='S'):

        # size [seq_len, label_len, pred_len]
        # info
        self.args = args

        self.insample_size = args.seq_len
        self.label_len = args.label_len
        self.outsample_size = args.pred_len

        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = None
        self.scale = True
        self.timeenc = 0
        
        # https://github.com/rakshitha123/TSForecasting
        file_name = os.path.join(root_path, args.dataset_name, args.dataset_name + ".tsf")
        df_loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(file_name)

        if frequency == "daily":
            self.freq = "d"
        
        print("frequency", frequency)

        """
        print(pd.unique(df_loaded_data.series_type))
        print(df_loaded_data[df_loaded_data.series_type=="solar"])
        print(df_loaded_data.columns)
        print(df_loaded_data.head)
        """
        if args.dataset_name == "weather_dataset":
            df_loaded_data = df_loaded_data[df_loaded_data.series_type==args.weather_type]

        num_series = 0
        all_lens = []
        for index, row in df_loaded_data.iterrows():
            x = row["series_value"]
            all_lens.append(len(x))
            num_series += 1
        min_len = min(all_lens)
        max_len = max(all_lens)

        print("[{}] Max/Minimum length: {}/{}".format(args.dataset_name, min_len, max_len))

        self.global_inx = 0
        self.num_vars = 0
        self.seq_i = {}
        self.seq_j = {}
        self.values = []
        self.lens = []
        for index, row in df_loaded_data.iterrows():
            len_ = len(row["series_value"])

            if args.dataset_name not in ['covid_deaths_dataset','dominick_dataset']:
                if len_ < 10000: # 2*(args.seq_len+args.pred_len):
                    continue
           
            start_pos = int(len_*0.8) # min(int(len_*0.6), len_-args.seq_len-args.pred_len)
            start_pos2 = int(len_*0.9) # min(int(len_*0.6), len_-args.seq_len-args.pred_len)
            
            border1s = [0, start_pos-args.seq_len,start_pos2-args.seq_len]
            border2s = [start_pos, start_pos2, len_]

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            """
            if len(row["series_value"][border1:border2]) == 0:
                print(index, len(row["series_value"][border1:border2]))
                raise Exception("")
            """

            self.scaler = StandardScaler()
            data_i = np.array(row["series_value"][border1:border2])
            if self.scale:
                # mean_ = torch.mean(x_enc[:,-48:,:], dim=1).unsqueeze(1)
                # std_ = torch.ones_like(torch.std(x_enc, dim=1).unsqueeze(1))

                # x_enc_i = (x_enc-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

                data_i = np.expand_dims(data_i, 1)
                self.scaler.fit(data_i)
                data_i = self.scaler.transform(data_i)
                data_i = np.squeeze(data_i, axis=1)

            plt.figure()
            plt.plot(data_i)
            plt.savefig("demo.png")

            self.values.append(data_i)
            self.lens.append(border2-border1)

            for t_i in range(border1,border2-args.seq_len-args.pred_len):

                self.seq_i[self.global_inx] = self.num_vars  
                self.seq_j[self.global_inx] = t_i-border1
                self.global_inx += 1

            self.num_vars += 1

        # print("self.values", len(self.values))
        print("[{}] After processing - Max/Minimum length: {}/{}".format(args.dataset_name, max(self.lens), min(self.lens)))

        self.dim_datetime_feats = -1
        self.data_x = self.values

    def __getitem__(self, index):

        index = random.randint(0,self.global_inx-1)

        insample = np.zeros((self.insample_size, 1))
        outsample = np.zeros((self.outsample_size+self.label_len, 1))

        var_i = self.seq_i[index]
        time_j = self.seq_j[index]

        # print(self.global_inx)
        # print("var_i", np.shape(self.values[var_i]), self.lens[var_i], time_j, time_j+self.insample_size+self.outsample_size, self.lens[var_i])
        timeseries = self.values[var_i][time_j:(time_j+self.insample_size+self.outsample_size)]
        # print("timeseries", np.shape(timeseries))

        insample_window = timeseries[:self.insample_size]
        outsample_window = timeseries[(self.insample_size-self.label_len):(self.insample_size+self.outsample_size)]
        insample[:,0] = insample_window # np.expand_dims(insample_window, 1)
        outsample[:,0] = outsample_window
        idx = var_i
        timestamps1 = np.asarray(list(range(time_j,time_j+self.insample_size)))
        timestamps2 = np.asarray(list(range(time_j+self.insample_size-self.label_len, time_j+self.insample_size+self.outsample_size)))

        """
        if np.shape(insample)[1] == 0:
            print(np.shape(insample), np.shape(outsample), var_i, time_j, np.shape(timeseries))
            raise Exception("")
        """
        # print(np.shape(insample), np.shape(outsample), np.shape(timestamps1), np.shape(timestamps2))
        return insample, outsample, np.zeros([np.shape(insample)[0], 1]), np.zeros([np.shape(outsample)[0], 1]), idx, timestamps1, timestamps2, self.lens[var_i]

    def __len__(self):
        if self.set_type == 0:
            return min(10000, self.global_inx)
        elif self.set_type == 2:
            return self.global_inx
        else:
            return self.global_inx

    


class Dataset_Stock(Dataset):
    """
    Daily financial dataset (log returns) with a *calendar-date* based split, e.g.

        train : 2008-01-01 ~ 2022-12-31
        val   : 2023-01-01 ~ 2023-12-31
        test  : 2024-01-02 ~ 2024-12-31

    The csv is expected to be:  date,<TICKER_1>,...,<TICKER_N>
    with each column already being r_close = log(adj_close / adj_close.shift(1)).

    As in the other loaders, the val/test windows are allowed to reach `seq_len` rows
    back into the previous period so that every window has a full history; only the
    prediction targets stay inside the val/test period.
    """

    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='stock_logret.csv',
                 target='OT', scale=True, timeenc=1, freq='b'):

        self.args = args

        if size == None:
            self.seq_len = 252
            self.label_len = 63
            self.pred_len = 21
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Training keeps the dense one-window-per-day sweep; val/test can step by
        # --eval_stride so that the windows do not overlap and every day is
        # forecast exactly once (see config.py).
        self.stride = 1
        if flag in ['val', 'test']:
            s = int(getattr(args, 'eval_stride', 1))
            self.stride = self.pred_len if s == 0 else max(s, 1)

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):

        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.sort_values('date').reset_index(drop=True)

        cols = [c for c in df_raw.columns if c != 'date']
        # keep `target` as the last column, so that the univariate metrics in
        # exp_main (preds[:, :, -1]) refer to the target series
        if self.target in cols:
            cols = [c for c in cols if c != self.target] + [self.target]
        df_raw = df_raw[['date'] + cols]

        dates = df_raw['date']

        def _range_idx(start, end, name):
            m = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            idx = np.where(m.values)[0]
            if len(idx) == 0:
                raise ValueError("No rows for the {} period {} ~ {} in {}".format(
                    name, start, end, self.data_path))
            return idx[0], idx[-1] + 1

        tr_s, tr_e = _range_idx(self.args.train_start, self.args.train_end, 'train')
        va_s, va_e = _range_idx(self.args.val_start, self.args.val_end, 'val')
        te_s, te_e = _range_idx(self.args.test_start, self.args.test_end, 'test')

        border1s = [tr_s, max(va_s - self.seq_len, 0), max(te_s - self.seq_len, 0)]
        border2s = [tr_e, va_e, te_e]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 1 and va_s - self.seq_len < 0:
            print("[warn] not enough history before the val period for seq_len={}".format(self.seq_len))
        if self.set_type == 2 and te_s - self.seq_len < 0:
            print("[warn] not enough history before the test period for seq_len={}".format(self.seq_len))

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        if self.timeenc == 0:
            df_stamp = df_stamp.copy()
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.dates = df_raw['date'].values[border1:border2]
        self.cols = list(df_data.columns)

        n_windows = len(self.data_x) - self.seq_len - self.pred_len + 1
        print("> {}-shape".format(self.flag), np.shape(self.data_x),
              "| rows {} ~ {} | windows: {}{}".format(
                  str(df_raw['date'].iloc[border1].date()),
                  str(df_raw['date'].iloc[border2 - 1].date()),
                  len(self), '' if self.stride == 1 else
                  ' (stride {}, non-overlapping; {} if dense)'.format(self.stride, n_windows)))
        if n_windows <= 0:
            raise ValueError(
                "seq_len({}) + pred_len({}) is longer than the {} period ({} rows). "
                "Use a shorter seq_len/pred_len.".format(
                    self.seq_len, self.pred_len, self.flag, len(self.data_x)))

    def __getitem__(self, index):

        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))

        return seq_x, seq_y, seq_x_mark, seq_y_mark, idx, timestamps1, timestamps2, len(self.data_x)

    def __len__(self):
        # only whole pred_len blocks are scored; a shorter tail cannot be stacked
        # into a batch (plot_price.py draws it separately)
        return (len(self.data_x) - self.seq_len - self.pred_len) // self.stride + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
