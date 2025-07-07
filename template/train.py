import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


from model import MyModel
from dataset import MyDataset


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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aaa", type=str, default="aaa", help="aaa")
    parser.add_argument("--bbb", type=int, default=666, help="666")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # load dataset
    train_dataset = MyDataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=1, shuffle=True)
    val_dataset = MyDataset()
    val_loader = DataLoader(val_dataset, batch_size=args.batch, num_workers=1, shuffle=False)

    # model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=...)


    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader,  optimizer, device)
        val_loss, val_mae = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d} | Train Loss {train_loss:.5f} | "
              f"Val Loss {val_loss:.5f} | Val MAE {val_mae:.5f}")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"epoch_{epoch}.pth")