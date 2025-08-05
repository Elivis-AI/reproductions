# [750, 850]
# [-173, 536]

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import MLP 
from dataset import build_dataset, MatDataset

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default="data/800s_001s_struct.mat",
                        help="新数据集 .mat 文件名")
    parser.add_argument("--batch", type=int, default=1,
                        help="推理时批大小")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    # 3.3 读取新数据集
    input_keys = [
        # 比赛没法用： 'VehVSpeedTarget', 'Grad_Road', 'I_Motor', 'P_Gen'
        # 相关性强： 'id', 'idRef', 'iq', 'VehV_v', 'MotorTrqTarget', 'MotorTrq', 'Uq'
        'Ud', 'Trq_Eng', 'TVA', 'SOC',
        'iqRef', 'nSped', 'n_Engine'
    ]
    output_keys = ['U_battery', 'I_Gen']
    dataset = MatDataset([args.mat], input_keys, output_keys)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    # 3.2 初始化模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=len(input_keys), act='leaky_relu').to(device)
    model.load_state_dict(torch.load("weights/tiny_mlp_epoch_300.pth", map_location=device))
    model.eval()

    # 3.5 推理
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            out = model(bx).cpu().numpy()
            preds.append(out)
    preds = np.vstack(preds).squeeze()


    # 3.7 绘图
    plt.figure(figsize=(10, 3))
    plt.plot(dataset.Y, label="Original ")
    plt.plot(preds, '--', label="Predicted")
    plt.xlabel("Sample Index (Full Set)")
    plt.ylabel("U_battery (V)")
    plt.title("Original vs Predicted (Full Set)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
