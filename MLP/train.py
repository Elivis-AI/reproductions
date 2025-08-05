import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from model import MLP
from dataset import build_dataset, MatDataset


# ------------------------------#
# 设置随机种子以确保可重复性
# ------------------------------#
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def cal_loss(pred, target):
    loss = F.mse_loss(pred, target, reduction='sum')
    # penalty_0 = torch.clamp(pred[:, 0] - 850, min=0) + torch.clamp(750 - pred[:, 0], min=0)
    # penalty_1 = torch.clamp(pred[:, 1] - d, min=0) + torch.clamp(-173 - pred[:, 1], min=0)
    # loss += (penalty_0 + penalty_1) * 100 * bx.size(0)
    return loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        pred = model(bx)
        loss = cal_loss(pred, by)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * bx.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, mae_sum = 0.0, 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        pred = model(bx)
        loss_sum += cal_loss(pred, by).item()
        mae_sum += torch.sum(torch.abs(pred - by)).item()
    return loss_sum / len(loader.dataset), mae_sum



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default="data/500s_001s_struct.mat", help=".mat 文件名")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch", type=int, default=1000, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="L2 权重衰减")
    args = parser.parse_args()

    # --------------------------#
    # load data
    # --------------------------#
    train_data_list = [
        "data/500s_001s_struct_1.mat",
        "data/500s_001s_struct_2.mat",
        "data/500s_001s_struct_3.mat",
    ]
    val_data_list = ['data/800s_001s_struct.mat']
    input_keys = [
        # 比赛没法用： 'VehVSpeedTarget', 'Grad_Road', 'I_Motor', 'P_Gen'
        # 相关性强： 'id', 'idRef', 'iq', 'VehV_v', 'MotorTrqTarget', 'MotorTrq', 'Uq'
        'Ud', 'Trq_Eng', 'TVA', 'SOC',
        'iqRef', 'nSped', 'n_Engine'
    ]
    output_keys = ['U_battery', 'I_Gen']

    train_dataset = MatDataset(train_data_list, input_keys, output_keys)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataset = MatDataset(val_data_list, input_keys, output_keys)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=len(input_keys), act='leaky_relu').to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)


    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader,  optimizer, device)
        val_loss, val_mae = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d} | Train Loss {train_loss:.5f} | "
              f"Val Loss {val_loss:.5f} | Val MAE {val_mae:.5f}")
        
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"weights/tiny_mlp_epoch_{epoch}.pth")

    # model.eval()

    # full_loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)
    # preds_norm_full, trues_norm_full = [], []
    # with torch.inference_mode():
    #     for bx, by in full_loader:
    #         bx = bx.to(device)
    #         pred_norm = model(bx).cpu().numpy()
    #         preds_norm_full.append(pred_norm)
    #         trues_norm_full.append(by.numpy())
    # preds_full = np.vstack(preds_norm_full)
    # trues_full = np.vstack(trues_norm_full)


    # plt.figure(figsize=(10, 3))
    # plt.plot(trues_full, label="Original U_battery")
    # plt.xlabel("Sample Index (Full Set)")
    # plt.ylabel("U_battery (V)")
    # plt.title("Original U_battery (Full Set)")
    # plt.legend()
    # plt.show()

    # # b) Full dataset - predicted only
    # plt.figure(figsize=(10, 3))
    # plt.plot(preds_full, '--', label="Predicted U_battery")
    # plt.xlabel("Sample Index (Full Set)")
    # plt.ylabel("U_battery (V)")
    # plt.title("Predicted U_battery (Full Set)")
    # plt.legend()
    # plt.show()

    # # c) Full dataset - overlay
    # plt.figure(figsize=(10, 3))
    # plt.plot(trues_full, label="Original ")
    # plt.plot(preds_full, '--', label="Predicted")
    # plt.xlabel("Sample Index (Full Set)")
    # plt.ylabel("U_battery (V)")
    # plt.title("Original vs Predicted (Full Set)")
    # plt.legend()
    # plt.show()