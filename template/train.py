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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, mae_sum = 0.0, 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        pred = model(bx)
        loss_sum += cal_loss(pred, by).item()
        # mae_sum += torch.sum(torch.abs(pred - by)).item()
    return loss_sum / len(loader.dataset) # , mae_sum


def train(args):
    # load dataset
    train_dataset = MyDataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=1, shuffle=True)
    val_dataset = MyDataset()
    val_loader = DataLoader(val_dataset, batch_size=args.batch, num_workers=1, shuffle=False)

    # model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # amp
    if args.amp and torch.cuda.is_available():
        ctx = torch.cuda.amp.autocast()
    else:
        from contextlib import nullcontext
        ctx = nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # frozen some parameters if needed
    for param in model.xxx_module.parameters():
        param.requires_grad = False

    model.train()
    running_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        for bx, by in train_loader:
            # forward
            bx, by = bx.to(device), by.to(device)
            with ctx:
                pred = model(bx)
                loss = cal_loss(pred, by)
                running_loss += loss.item()
            
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d} | Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f}")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"epoch_{epoch}.pth")



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decays")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping")

    parser.add_argument("--amp", type=bool, default=True, help="use automatic mixed precision")
    parser.add_argument("--dtype", type=str, choices=['float32', 'float16', 'bfloat16'], default='float32', help="data type for training")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)

        
