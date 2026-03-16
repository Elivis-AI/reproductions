from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ##################################################################################
#   DllmMixin: 扩散语言模型的核心通用逻辑
#   包含: 双向注意力改造 / 噪声注入 / 加权损失 / Block-wise 生成
# ##################################################################################

class DllmMixin:

    # ------------------------------------------------------------------
    #  双向注意力改造
    # ------------------------------------------------------------------
    def _bidirectional_mask(self, attention_mask, input_tensor, *args, **kwargs):
        """
        双向注意力 mask: 仅遮蔽 padding token。
        返回 shape: [bsz, 1, seq_len, seq_len]
        """
        bsz, seq_len = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        if attention_mask is not None:
            expanded = attention_mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len).to(dtype)
            return (1.0 - expanded) * torch.finfo(dtype).min

        return torch.zeros(bsz, 1, seq_len, seq_len, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    #  MDM-style noise (token masking)
    # ------------------------------------------------------------------
    def add_noise_to_tokens(
            self, 
            input_ids: torch.LongTensor, 
            eps: float, 
            pad_token_id: int,
            mask_token_id: int
        ):
        """
        MDM-style 前向噪声过程: 随机将部分 token 替换为 [MASK]。

        流程:
          1. 为 batch 中每个样本采样噪声水平 t ~ U(0, 1)
          2. 计算 mask 概率 p = (1 - ε)t + ε, 保证 p ∈ [ε, 1]
          3. 每个位置以概率 p 独立替换为 [MASK], padding 位置不替换

        返回:
          noisy_input_ids:  加噪后的 input_ids, 部分 token 变为 mask_token_id
          corruption_mask:  bool, 标记哪些位置被替换 [bsz, seq_len]
          p_mask:           每个位置的 mask 概率, 用于加权损失 [bsz, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        t = torch.rand(input_ids.shape[0], device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask.unsqueeze(-1).expand(batch_size, seq_len)
        corruption_mask = torch.rand(batch_size, seq_len, device=input_ids.device) < p_mask
        corruption_mask = corruption_mask & (input_ids != pad_token_id)
        noisy_input_ids = torch.where(corruption_mask, mask_token_id, input_ids)
        return noisy_input_ids, corruption_mask, p_mask

    # ------------------------------------------------------------------
    #  加权交叉熵损失
    # ------------------------------------------------------------------
    def dllm_loss(self, flatten_logits, flatten_labels, corruption_mask, p_mask):
        """DLLM 加权交叉熵: loss[mask] / p_mask[mask]"""
        ce = F.cross_entropy(
            flatten_logits,
            flatten_labels,
            reduction="none",
        ).view(flatten_labels.shape)
        return (ce[corruption_mask] / p_mask[corruption_mask]).sum() / flatten_labels.numel()


    # ------------------------------------------------------------------
    #  Block-wise 迭代去噪生成
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        steps: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """DLLM Block-wise 生成: 每次生成一个 block，逐块去噪"""
        input_ids = kwargs.get("input_ids", inputs)
        bsz, prompt_len, device = input_ids.shape[0], input_ids.shape[1], input_ids.device

        mask_id = mask_token_id if mask_token_id is not None else getattr(self, "mask_token_id", None)
        if mask_id is None:
            raise ValueError("mask_token_id 未提供且模型未设置 self.mask_token_id")

        eos_id = eos_token_id if eos_token_id is not None else getattr(self.config, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("eos_token_id 未提供且 self.config.eos_token_id 未设置")

        block_size = kwargs.get("block_size", 16)
        steps_per_block = steps or 8

        x = input_ids.clone()
        generated_len = 0

        while generated_len < max_new_tokens:
            cur_block = min(block_size, max_new_tokens - generated_len)
            block_mask = torch.full((bsz, cur_block), mask_id, dtype=torch.long, device=device)
            x = torch.cat([x, block_mask], dim=1)
            block_start = x.shape[1] - cur_block

            for step in range(steps_per_block):
                mask_pos = (x[:, block_start:] == mask_id)
                if not mask_pos.any():
                    break

                logits = self(input_ids=x).logits[:, block_start:, :] / temperature
                if top_k > 0:
                    logits[logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(bsz, -1)
                conf = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
                conf = torch.where(mask_pos, conf, -1.0)

                n_fill = cur_block if step == steps_per_block - 1 else max(1, cur_block // steps_per_block)
                _, top_idx = conf.topk(min(n_fill, mask_pos.sum(-1).min().item()), dim=-1)
                for b in range(bsz):
                    x[b, block_start + top_idx[b]] = sampled[b, top_idx[b]]

            generated_len += cur_block

            if tokenizer and kwargs.get("stream", False):
                text = tokenizer.decode(x[0, prompt_len:], skip_special_tokens=False)
                print(f"[Block {generated_len // block_size}] {text}")

            if eos_id is not None and (x[:, prompt_len:] == eos_id).any(dim=-1).all():
                break

        if eos_id is not None:
            for b in range(bsz):
                eos_pos = (x[b, prompt_len:] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    x = x[:, : prompt_len + eos_pos[0].item() + 1]
                    break
        return x

