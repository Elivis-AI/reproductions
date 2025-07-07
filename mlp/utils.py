import numpy as np

def minmax_scaler(x: np.ndarray):
    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))


def diff(x):
    N, D = x.shape
    return np.vstack([np.zeros((1, D), dtype=x.dtype), np.diff(x, axis=0)])