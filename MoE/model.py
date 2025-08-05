import torch 
import torch.nn as nn
import torch.nn.functional as F

from config import MoEConfig
from dataset import CIFAR10


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(dim, 2* dim),
            nn.ReLU(),
            nn.Linear(2* dim, dim),
        )

    def forward(self,x):
        return self.model(x)



class MoE(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.alpha = config.alpha
        self.infer_acc = config.infer_acc
        
        self.experts = nn.ModuleList([Expert(config.dim) for _ in range(config.n_experts)])
        self.gate = nn.Linear(config.dim, self.n_experts, bias=False)
        
        self.init_parameters()

    def init_parameters(self) -> None:
        # Ref: minimind 
        nn.init.kaiming_uniform_(self.gate.weight, 2.2361) # sqrt(5)
        
    def forward(self, x: torch.Tensor):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # 使用门控机制选择专家
        logits = self.gate(x)
        p_experts = F.softmax(logits, dim=-1) # (B, L, p)

        # 选择 top_k 个专家
        # topk_p, topk_idx: (B, L, top_k)
        topk_p, topk_idx = torch.topk(p_experts, k=self.top_k, dim=-1, sorted=False)
        # 对 top_k 权重进行归一化
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True) + 1e-20     
        

        # 辅助损失
        # Ref: https://arxiv.org/pdf/2101.03961
        ce = F.one_hot(topk_idx.view(-1), num_classes=self.n_experts).float().mean(0)
        Pi = p_experts.mean(0)
        fi = ce * self.n_experts
        aux_loss = (Pi * fi).sum() * self.alpha

        # 训练模式下，将每个token输入到topk个专家中
        x = x.repeat_interleave(self.top_k, dim=0).reshape(-1, D) # (B * L * top_k, D)
        y = torch.empty_like(x)
        topk_idx = topk_idx.view(-1) # (top_k * B * L)
        for i, expert in enumerate(self.experts):
            y[topk_idx == i] = expert(x[topk_idx == i])
        y = y.reshape(B, self.top_k, L, D).permute(0, 2, 1, 3) # (B, L, top_k, D)
        y = (y * topk_p.unsqueeze(-1)).sum(dim=2)

        # elif self.infer_acc is True:
        #     # 测试模式下，选择top1
        #     top1_idx = torch.argmax(p_experts, dim=-1) # (B, L)
            
        #     x = x.reshape(-1, D)
        #     y = torch.empty_like(x)
        #     top1_idx = top1_idx.view(-1) # (B * L)
        #     for i, expert in enumerate(self.experts):
        #         y[top1_idx == i] = expert(x[top1_idx == i])
        #     y = y.reshape(B, L, D)
        
        return y, ce, aux_loss


# MNIST分类
class MNIST_MoE(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.emb = nn.Conv2d(
            in_channels = 1,
            out_channels = config.dim,
            kernel_size = 7,
            padding = 0,
            stride = 7,
        ) # turn 28*28 image into 4 * 4 blocks
        self.moe = MoE(config)
        self.cls = nn.Linear(16 * config.dim, 10)
        
    def forward(self, x: torch.Tensor):
        x = self.emb(x)
        x = x.flatten(2).transpose(2, 1)
        y, ce, aux_loss = self.moe(x)
        y = y.flatten(1)  # (B, L, D) -> (B, L * D)
        return self.cls(y), ce, aux_loss



if __name__=='__main__':
    config = MoEConfig()
    moe=MoE(config)
    x=torch.rand((5, 10, config.dim))
    y, ce, aux_loss=moe(x)

    print(y.shape)
    print(ce)
    print(aux_loss)