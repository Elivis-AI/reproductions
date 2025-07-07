import torch
import numpy as np
from torch.utils.data import Dataset

from scipy.io import loadmat
from utils import minmax_scaler
from scipy.ndimage import gaussian_filter1d

ALL_KEYS = {
    'VehVSpeedTarget', 'Grad_Road', 'I_Motor', 'P_Gen'
    'id', 'idRef', 'iq', 'VehV_v', 'MotorTrqTarget', 'MotorTrq', 'Uq'
    'Ud', 'Trq_Eng', 'TVA', 'SOC', 'iqRef', 'nSped', 'n_Engine',
    'U_battery', 'I_Gen'
}

# 定值缩放
MAP_X = {
    'Ud': lambda x: x / 200, 
    'Trq_Eng': lambda x: x / 3000, 
    'TVA': lambda x: x / 100, 
    'SOC': lambda x: x,
    'iqRef': lambda x: x / 150, 
    'nSped': lambda x: x / 800, 
    'n_Engine': lambda x: x / 3000,
}

def build_dataset(path_list, x_keys, y_keys):
    x_keys, y_keys = set(x_keys), set(y_keys)
    # assert x_keys.issubset(ALL_KEYS) and y_keys.issubset(ALL_KEYS), \
    #     f"Keys must be a subset of {ALL_KEYS}, got x_keys: {x_keys}, y_keys: {y_keys}"
    
    # load data from multiple sources
    data_list = [loadmat(path) for path in path_list]
    data = {
        k: np.vstack([d[k] for d in data_list]) \
            for k in data_list[0].keys() if isinstance(data_list[0][k], np.ndarray)
    }
    data['Ud'] = gaussian_filter1d(data['Ud'][:,0], sigma=3).reshape(-1, 1) # 对抖动过大的数据做平滑处理
    
    assert x_keys.issubset(MAP_X.keys())
    X = np.hstack([MAP_X[k](data[k]) for k in x_keys]).astype(np.float32)
    Y = np.hstack([data[k] for k in y_keys]).astype(np.float32)
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


class MatDataset(Dataset):
    def __init__(self, path_list, x_keys, y_keys):
        super().__init__()
        self.X, self.Y = build_dataset(path_list, x_keys, y_keys)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

class SlideWindowDataset(Dataset):
    def __init__(self, path_list, x_keys, y_keys, window_size):
        super().__init__()
        self.window_size = window_size
        self.X, self.Y = build_dataset(path_list, x_keys, y_keys)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx-self.window_size:idx], self.Y[idx-self.window_size:idx]
