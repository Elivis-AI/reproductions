
import os
from datetime import timedelta

import torch
import torch.nn as nn

class EMALoss():
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss = None

    def reset(self):
        self.loss = None

    def update(self, loss):
        if self.loss is None:
            self.loss = loss.item()
        else:
            self.loss = self.alpha * loss.item() + (1 - self.alpha) * self.loss
    
    def item(self):
        return self.loss

def print_num_parameters(model: nn.Module):
    print(f'参数量：{sum(p.numel() for p in model.parameters()) / 1e9:.3f} B')
    
def print_num_trainable_parameters(model: nn.Module):
    print(f'可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.3f} B')

#################################################################################
#                              distributed utils                                #
#################################################################################

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif (
        "SLURM_PROCID" in os.environ and os.environ["SLURM_JOB_NAME"] != "bash"
    ):  # Exclude interactive shells
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, 'dist_on_itp') and args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(hours=1),
    )
    torch.distributed.barrier()
