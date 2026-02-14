import os
import random
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler, DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer

from pt_model import MyGPT
from pt_config import ModelConfig
from dataset import JsonPretrainDataset

import warnings
warnings.filterwarnings('ignore')


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########## distributed utils ##########

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main_process_print(content):
    if is_main_process():
        print(content)

def init_dist():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# def get_model_params(model, config):
#     total = sum(p.numel() for p in model.parameters()) / 1e6
#     n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
#     n_active = getattr(config, 'num_experts_per_tok', 0)
#     n_shared = getattr(config, 'n_shared_experts', 0)
#     expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
#     shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
#     base = total - (expert * n_routed) - (shared_expert * n_shared)
#     active = base + (expert * n_active) + (shared_expert * n_shared)
#     if active < total: main_process_print(f'Model Params: {total:.2f}M-A{active:.2f}M')
#     else: main_process_print(f'Model Params: {total:.2f}M')

########## data utils ##########

def pretrain_collate_fn(batch, pad_token_id=0):
    input_ids = [item['input_ids'] for item in batch]
    max_len = max(ids.size(0) for ids in input_ids)
    input_ids = torch.stack([
        F.pad(ids, (0, max_len - ids.size(0)), value=pad_token_id)
        for ids in input_ids
    ])
    return {'input_ids': input_ids}


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


def train(args):

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_dist()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 定义模型、数据、优化器 ==========
    lm_config = ModelConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=6400,
        max_seq_len=args.max_seq_len,
        padding_idx=None,
        rope_theta=1e6,
        dropout=0.0,
        flash_attn=True
    )
    model = MyGPT(lm_config).to(args.device)
    if args.weight:
        weights = torch.load(args.weight, map_location=args.device)
        model.load_state_dict(weights, strict=False)
    assert args.tokenizer or args.weight
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.weight)
    tokenizer.model_max_length = args.max_seq_len

    train_ds = JsonPretrainDataset(args.data_path, tokenizer)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6) # lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))
    
    # ========== 3. 加载状态（断点续训） ==========
    start_epoch, start_step = 0, 0
    if args.resume:
        main_process_print(f'Resuming from {args.resume}')

        resume_path = os.path.join(args.resume, 'ckpt.pt')
        assert os.path.exists(resume_path)
        ckpt_data = torch.load(resume_path, map_location='cpu')

        model.load_state_dict(ckpt_data['model'])
        optimizer.load_state_dict(ckpt_data['optimizer'])
        scheduler.load_state_dict(ckpt_data['scheduler'])
        scaler.load_state_dict(ckpt_data['scaler'])
        start_epoch = ckpt_data['epoch']
        start_step = ckpt_data.get('step', 0)
        
        # GPU数量变化时，自动转换step
        saved_ws = ckpt_data.get('world_size', 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1
        if saved_ws != current_ws:
            start_step = start_step * saved_ws // current_ws
    
    # ========== 4. 设置DDP、混合精度、编译模型 ==========
    if args.use_compile:
        model = torch.compile(model)
        main_process_print('torch.compile enabled')

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if "cuda" in args.device else nullcontext()

    # ========== 5. 配置logger(wandb) ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckpt_data.get('wandb_id') if args.resume else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.lr}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 6. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # prepare dataloader
        # Dataset      :   提供 `__getitem__(idx)`
        # Sampler      :   生成 idx 序列
        # BatchSampler :   把 idx 组成 batch
        # collate_fn   :   把 batch 数据整理成模型输入格式
        # DataLoader   :   根据 idx 真正取数据
        train_sampler and train_sampler.set_epoch(epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        collate_fn = lambda batch: pretrain_collate_fn(batch, pad_token_id=tokenizer.pad_token_id)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        
        start_time = time.time()
        iters = len(loader) + skip
        start = start_step if skip > 0 else 0
        # one epoch
        for step, input in enumerate(loader, start=start):
            input_ids = input['input_ids'].to(args.device)
            # lr = get_lr(epoch * iters + step, args.epochs * iters, args.lr)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr

            with autocast_ctx:
                output = model(input_ids)
                logits = output['logits']
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1), reduction='mean')
                loss = loss / args.accumulation_steps

            loss = scaler.scale(loss)
            loss.backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer) # optimizer.step() 的scaler版本
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step % args.log_interval == 0 or step == iters - 1:
                spend_time = time.time() - start_time
                current_loss = loss.item() * args.accumulation_steps
                current_lr = optimizer.param_groups[-1]['lr']
                eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
                main_process_print(
                    f"Epoch [{epoch + 1}/{args.epochs}]  Step [{step}/{iters}]\n"
                    f"  loss        : {current_loss:.4f}\n"
                    f"  lr          : {current_lr:.8f}\n"
                    f"  epoch_time  : {eta_min:.1f} min\n"
                )
                if wandb:
                    wandb.log(
                        {
                            "loss": current_loss, 
                            "lr": current_lr, 
                            "epoch_time": eta_min
                        }
                    )

            if is_main_process() and (step % args.save_interval == 0 or step == iters - 1):
                os.makedirs(args.save_dir, exist_ok=True)

                raw_model = model.module if isinstance(model, DistributedDataParallel) else model # 去掉DDP包装
                raw_model = getattr(raw_model, '_orig_mod', raw_model) # 去掉torch.compile包装
                state_dict = raw_model.state_dict()
                
                # save weights (fp16)
                weight_path = os.path.join(args.save_dir, f'weight.pt')
                torch.save({k: v.half().cpu() for k, v in state_dict.items()}, weight_path)
                weight_tmp = weight_path + '.tmp' # safe save
                torch.save(state_dict, weight_tmp)
                os.replace(weight_tmp, weight_path)
                
                if wandb:
                    if hasattr(wandb, 'get_run'):
                        run = wandb.get_run()
                        wandb_id = getattr(run, 'id', None) if run else None
                    else:
                        wandb_id = getattr(wandb, 'id', None)
                else:
                    wandb_id = None

                # save checkpoint
                resume_path = os.path.join(args.save_dir, f'checkpoint.pth')
                resume_data = {
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'world_size': dist.get_world_size() if dist.is_initialized() else 1,
                    'wandb_id': wandb_id
                }
                resume_tmp = resume_path + '.tmp'
                torch.save(resume_data, resume_tmp)
                os.replace(resume_tmp, resume_path)

                # clean
                del state_dict, resume_data
                torch.cuda.empty_cache()

            del input_ids, output, loss    

    # ========== 7. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraining Script")
    parser.add_argument("--save_dir", type=str, default="./results", help="模型保存目录")
    parser.add_argument('--tokenizer', type=str, help="tokenizer路径")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', action="store_true", help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="./data/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--weight', type=str, help="基于训练的权重路径")
    parser.add_argument('--resume', type=str, help="断点恢复的路径")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", action="store_true", help="是否使用torch.compile加速")
    args = parser.parse_args()

    train(args)