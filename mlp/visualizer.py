import numpy as np
import matplotlib.pyplot as plt

from dataset import build_dataset
from utils import minmax_scaler

def plot_high_dim_series(data, key):
    """
    可视化高维数据中每个维度分量的变化趋势（归一化后）。
    
    """
    data = np.asarray(data)

    N, D = data.shape
    
    plt.figure(figsize=(12, 6))
    x = np.arange(N)

    for d in range(D):
        plt.plot(x, minmax_scaler(data[:, d]), label=key[d])

    # plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_subplots(X, key):
    """
    将每个维度绘制为一个单独的子图
    """
    N , D = X.shape
    
    # 设置图像排布：6 行 3 列
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    t = np.arange(N)

    for d in range(D):
        ax = axes[d // 3, d % 3]
        ax.plot(t, X[:, d], color='C0')
        # ax.plot(t, dX[:, d], color='C1')
        ax.set_title(key[d])
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.subplots_adjust(
        top=0.92,   # ⬅️ 适当压缩子图顶部区域，给标题留空间
        bottom=0.05,
        left=0.05,
        right=0.95,
        hspace=0.5,
        wspace=0.3
    )

    plt.subplots_adjust()
    plt.show()

# def smooth(x, window_size):
#     if window_size < 1 or window_size % 2 == 0:
#         raise ValueError("window_size 必须为正奇数")

#     # 使用 np.convolve 进行滑动平均
#     kernel = np.ones(window_size) / window_size
#     return np.convolve(x, kernel, mode='same')  # 'same' 保持长度一致

def smooth_curve_moving_average(data, window_size=10):
    """
    对 (N, 1) 形状的数据使用滑动平均进行平滑处理。
    
    参数：
        data: 形状为 (N, 1) 的 NumPy 数组
        window_size: 滑动窗口大小，必须为正奇数，控制平滑程度
    
    返回：
        平滑后的 (N, 1) NumPy 数组
    """
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 1:
        raise ValueError("输入必须是 (N, 1) 的二维数组")

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data[:, 0], kernel, mode='same')
    return smoothed.reshape(-1, 1)


data_path_list = [
    "data/500s_001s_struct_1.mat",
    "data/500s_001s_struct_2.mat",
    "data/500s_001s_struct_3.mat",
]

input_keys = [
    'Ud', 'Trq_Eng', 'TVA', 'SOC',
    'iqRef', 'nSped', 'n_Engine'
] 
# 比赛没法用： 'VehVSpeedTarget', 'Grad_Road', 'I_Motor', 'P_Gen'
# 相关性强： 'id', 'idRef', 'iq', 'VehV_v', 'MotorTrqTarget', 'MotorTrq', 'Uq'



output_keys = ['U_battery', 'I_Gen']
keys = input_keys + output_keys
X, Y = build_dataset(data_path_list, input_keys, output_keys)
all = np.hstack([X, Y]).astype(np.float32)
# Y = np.hstack([data['U_battery'], data['I_Gen']]).astype(np.float32)  # (N, 2)
# dAll = diff(all) / 0.01
# plot_high_dim_series(X, input_keys)
# plot_high_dim_series(Y, output_keys)
# plot_subplots(X, input_keys)
plot_subplots(all, keys)
