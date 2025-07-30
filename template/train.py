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
    # loss = F.mse_loss(pred, target, reduction='sum')
    # return loss
    pass


@torch.no_grad()
def evaluate(model, loader, device):
    # model.eval()
    # loss_sum, mae_sum = 0.0, 0.0
    # for bx, by in loader:
    #     bx, by = bx.to(device), by.to(device)
    #     pred = model(bx)
    #     loss_sum += cal_loss(pred, by).item()
    #     # mae_sum += torch.sum(torch.abs(pred - by)).item()
    # return loss_sum / len(loader.dataset) # , mae_sum
    pass


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
        precision_context = torch.cuda.amp.autocast()
    else:
        from contextlib import nullcontext
        precision_context = nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # frozen some parameters if needed
    # for param in model.xxx_module.parameters():
    #     param.requires_grad = False

    print(f'参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    print(f'可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M')

    model.train()
    running_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        for bx, by in train_loader:
            # forward
            bx, by = bx.to(device), by.to(device)
            with precision_context:
                pred = model(bx)
                loss = cal_loss(pred, by)
                running_loss += loss.item()
            
            ### vanilla backpropgation
            # optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # optimizer.step()

            ## backpropgation with AMP
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer) # replace optimizer.step()
            scaler.update()

            ### backward with grad accumulation
            # loss.backward()
            # step += 1
            # if (step + 1) % accumulation_steps == 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            #     optimizer.step()
            #     optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d} | Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f}")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"epoch_{epoch}.pth")


def ddp_train(args):
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DistributedSampler
    from utils import init_distributed_mode, get_rank
    
    init_distributed_mode(args)

    # load dataset
    train_dataset = MyDataset()
    # A. add distributed sampler
    train_sampler = DistributedSampler(train_dataset) # 在分布式训练中，确保每个 GPU 只处理它那一份数据，不会重复、也不会遗漏
    train_loader = DataLoader(
        train_dataset,                # 1. 训练数据集对象（Dataset）
        batch_size=args.batch_size,   # 2. 每个 batch 中的样本数量
        pin_memory=True,              # 3. 是否将数据预加载到 CUDA 固定内存
        drop_last=False,              # 4. 是否丢弃最后不足 batch_size 的 batch
        shuffle=False,                # 5. 是否打乱数据（注意：使用 sampler 时应设为 False）
        num_workers=args.num_workers, # 6. 使用多少个线程/进程加载数据
        sampler=train_sampler         # 7. 数据采样器（指定如何分配数据给进程）
    )
    
    # model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    # B. add DDP model wrapper
    model = DistributedDataParallel(model, device_ids=[get_rank()])
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

    print(f'参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    print(f'可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M')

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
            
            ### vanilla backpropgation
            # optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # optimizer.step()

            ## backpropgation with AMP
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer) # replace optimizer.step()
            scaler.update()

            ### backward with grad accumulation
            # loss.backward()
            # step += 1
            # if (step + 1) % accumulation_steps == 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            #     optimizer.step()
            #     optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:3d} | Train Loss {train_loss:.5f}")
        
        if epoch % 10 == 0:
            # assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
            # C. use `model.module` to access the underlying model
            torch.save(model.module.state_dict(), f"epoch_{epoch}.pth")


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

        
