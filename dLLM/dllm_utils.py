import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ##################################################################################
#   DllmMixin: 扩散语言模型的核心通用逻辑
#   包含: 双向注意力改造 / 噪声注入 / 加权损失 / Block-wise 生成
# ##################################################################################

class DllmMixin:
    """
    dLLM 核心 Mixin，宿主类需满足:
      - self.model: 含 .layers (nn.ModuleList) 的 Transformer backbone
      - self.lm_head: 输出投影层
      - self.config: 含 vocab_size, eos_token_id
      - self.mask_token_id: int
      - self.mask_epsilon: float
    """

    # ------------------------------------------------------------------
    #  双向注意力改造
    # ------------------------------------------------------------------
    def _make_bidirectional(self):
        """移除 causal mask，使所有 attention 层变为双向"""
        for layer in self.model.layers:
            layer.self_attn.is_causal = False
        self.model._update_causal_mask = self._bidirectional_mask

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
    def add_noise_to_tokens(self, input_ids: torch.LongTensor, t: torch.FloatTensor,
                            eps: float = None, pad_token_id: int = 0):
        """
        根据噪声水平 t 将 token 替换为 [MASK]。
        p_mask = (1 - eps) * t + eps,  t=0→eps, t=1→1
        """
        batch_size, seq_len = input_ids.shape
        eps = eps if eps is not None else self.mask_epsilon
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask.unsqueeze(-1).expand(batch_size, seq_len)
        corruption_mask = torch.rand(batch_size, seq_len, device=input_ids.device) < p_mask
        corruption_mask = corruption_mask & (input_ids != pad_token_id)
        noisy_input_ids = torch.where(corruption_mask, self.mask_token_id, input_ids)
        return noisy_input_ids, corruption_mask, p_mask

    # ------------------------------------------------------------------
    #  加权交叉熵损失
    # ------------------------------------------------------------------
    def dllm_loss(self, logits, labels, corruption_mask, p_mask):
        """DLLM 加权交叉熵: loss[mask] / p_mask[mask]"""
        ce = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            reduction="none",
        ).view(labels.shape)
        return (ce[corruption_mask] / p_mask[corruption_mask]).sum() / labels.numel()


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

