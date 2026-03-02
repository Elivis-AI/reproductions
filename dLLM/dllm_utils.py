import math
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput


# ##################################################################################
#   dLLM Generation Mixin: Block-wise 迭代去噪生成
# ##################################################################################

class DLLMGenerationMixin:

    @torch.inference_mode()
    def generate(
        self,
        inputs = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        steps: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """DLLM Block-wise 生成: 每次生成一个 block，逐块去噪"""
        input_ids = kwargs.get("input_ids", inputs)
        bsz, prompt_len, device = input_ids.shape[0], input_ids.shape[1], input_ids.device

        # mask_token_id: 优先使用传入值，否则读取模型属性
        mask_id = mask_token_id if mask_token_id is not None else getattr(self, "mask_token_id", None)
        if mask_id is None:
            raise ValueError("mask_token_id 未提供且模型未设置 self.mask_token_id")

        # eos_token_id: 优先使用传入值，否则读取 config
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

        # 截断到第一个 EOS
        if eos_id is not None:
            for b in range(bsz):
                eos_pos = (x[b, prompt_len:] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    x = x[:, : prompt_len + eos_pos[0].item() + 1]
                    break
        return x


# ##################################################################################
#   dLLM Data Collator: 在 collate 阶段注入噪声
# ##################################################################################

class DLLMDataCollator:
    """
    1. 对 batch 进行动态 padding
    2. 为每个样本采样噪声水平 t ~ U(0,1)
    3. 根据 t 将部分 token 替换为 [MASK]
    4. 返回 noisy input_ids, labels (clean), corruption_mask, p_mask
    """

    def __init__(
        self,
        tokenizer: Any,
        mask_token_id: int,
        mask_epsilon: float = 0.001
    ) -> None:
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.mask_epsilon = mask_epsilon

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. 动态 padding
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"]} for f in features],
            return_tensors="pt",
            padding=True,
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()

        bsz, seq_len = input_ids.shape
        eps = self.mask_epsilon

        # 2. 采样噪声水平 t ~ Uniform(0, 1)
        t = torch.rand(bsz)

        # 3. 计算每个位置的 mask 概率: p = (1 - eps) * t + eps
        p_mask = ((1 - eps) * t + eps).unsqueeze(-1).expand(bsz, seq_len).contiguous()

        # 4. 按概率替换为 [MASK]
        corruption_mask = torch.rand(bsz, seq_len) < p_mask
        corruption_mask = corruption_mask & attention_mask.bool()  # 不 mask padding

        noisy_input_ids = torch.where(corruption_mask, self.mask_token_id, input_ids)

        return {
            "input_ids": noisy_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "corruption_mask": corruption_mask,
            "p_mask": p_mask,
        }
