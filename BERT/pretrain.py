import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from utils import init_distributed_mode, get_rank, is_main_process

from transformers import RobertaTokenizerFast
from model import RoBERTaForPretraining
from mlm import MaskedLanguageModel
from dataset import BookCorpus
from utils import print_num_parameters
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

torch.autograd.set_detect_anomaly(True)

def apply_tokenizer(tokenizer):
    def func(texts):
        return tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    return func


def train(args):
    init_distributed_mode(args)

    # load dataset
    train_dataset = BookCorpus(file=args.data)
    # A. add distributed sampler
    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
    train_sampler = DistributedSampler(train_dataset) # 在分布式训练中，确保每个 GPU 只处理它那一份数据，不会重复、也不会遗漏
    train_loader = DataLoader(
        train_dataset,                # 1. 训练数据集对象（Dataset）
        batch_size=args.batch,   # 2. 每个 batch 中的样本数量
        pin_memory=True,              # 3. 是否将数据预加载到 CUDA 固定内存
        drop_last=False,              # 4. 是否丢弃最后不足 batch_size 的 batch
        shuffle=False,                # 5. 是否打乱数据（注意：使用 sampler 时应设为 False）
        num_workers=args.num_workers, # 6. 使用多少个线程/进程加载数据
        sampler=train_sampler,         # 7. 数据采样器（指定如何分配数据给进程）
        collate_fn=apply_tokenizer(tokenizer)
    )
    
    # model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlm = MaskedLanguageModel(
        vocab_size=tokenizer.vocab_size,
        mask_id=tokenizer.mask_token_id,
        no_mask_token_ids=[
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
        ],
    )
    model = RoBERTaForPretraining(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.model_max_length,
        padding_idx=tokenizer.pad_token_id,
        n_layers=12,
        n_heads=12,
        dim=768,
    ).to(device)

    # B. add DDP model wrapper
    model: nn.parallel.DistributedDataParallel = DistributedDataParallel(model, device_ids=[get_rank()])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # amp
    if args.amp and torch.cuda.is_available():
        ctx = torch.cuda.amp.autocast()
    else:
        from contextlib import nullcontext
        ctx = nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    if is_main_process():
        print_num_parameters(model)

    model.train()

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch in train_loader:
            # forward
            bx = batch.input_ids  # (B, L)
            pad_mask = batch.attention_mask

            bx = bx.to(device)
            bx, mask = mlm(bx)
            with ctx:
                pred = model(bx, pad_mask)
                loss = mlm.cal_loss(pred, bx, mask)
                running_loss += loss.item()

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
        
        if is_main_process():
            print(f"Epoch {epoch:3d} | Train Loss {train_loss:.5f}")
            
            if epoch % 10 == 0:
                # assert isinstance(model, nn.parallel.DistributedDataParallel)
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

    parser.add_argument("--data", type=str)
    parser.add_argument("--num-workers", type=int, default=1, help="number of workers for data loading")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)

        
