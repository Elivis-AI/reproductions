import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.modeling_outputs import MaskedLMOutput

from dllm_utils import DLLMGenerationMixin, DLLMDataCollator
from dataset import JsonPretrainDataset

import warnings
warnings.filterwarnings("ignore")


@dataclass
class QwenDLLMTrainingArguments:
    # 模型路径 (Qwen2 预训练检查点)
    model_name_or_path: str = field(metadata={"help": "Qwen2 预训练模型路径, 如 Qwen/Qwen2.5-0.5B"})

    # 数据参数
    dataset_path: str = field(metadata={"help": "训练数据路径 (jsonl 格式, 每行包含 'text' 字段)"})

    # dLLM 参数
    max_seq_len: int = field(default=512, metadata={"help": "最大序列长度"})
    mask_epsilon: float = field(default=0.001, metadata={"help": "最小 mask 概率 (噪声调度下界)"})


# ##################################################################################
#   Qwen2 → dLLM: 将 Qwen2 自回归模型转换为双向 Masked Diffusion Language Model
#   参照 model.py 中 MiniMindForMaskedDiffusion 的设计
# ##################################################################################

class QwenForMaskedDiffusion(DLLMGenerationMixin, Qwen2PreTrainedModel):
    """
    基于 Qwen2 架构的扩散语言模型 (dLLM)。
    核心修改:
      1. 注意力从 causal (单向) 改为 bidirectional (双向)
      2. 添加 MDM-style noise schedule 和 token masking
      3. 使用加权交叉熵损失: loss[mask] / p_mask[mask]
      4. Block-wise 迭代去噪生成
    """
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # dLLM 特有配置
        self.mask_token_id = getattr(config, "mask_token_id", config.vocab_size - 1)
        self.mask_epsilon = getattr(config, "mask_epsilon", 0.001)

        # 将注意力改为双向
        self._make_bidirectional()
        self.post_init()

    # ------------------------------------------------------------------
    #  权重绑定 (与 Qwen2ForCausalLM 一致)
    # ------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # ------------------------------------------------------------------
    #  双向注意力
    # ------------------------------------------------------------------
    def _make_bidirectional(self):
        """移除 causal mask，使所有 attention 层变为双向"""
        # 1. 逐层设置 is_causal = False
        for layer in self.model.layers:
            layer.self_attn.is_causal = False

        # 2. 覆盖 _update_causal_mask，返回双向 mask (仅遮蔽 padding)
        self.model._update_causal_mask = self._bidirectional_mask

    def _bidirectional_mask(self, attention_mask, input_tensor, *args, **kwargs):
        """
        生成双向注意力 mask:
          - 有 attention_mask 时: 仅遮蔽 padding token
          - 无 attention_mask 时: 返回全零 (全部 attend)
        返回 shape: [bsz, 1, seq_len, seq_len]
        """
        bsz, seq_len = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        if attention_mask is not None:
            # attention_mask: [bsz, seq_len], 1=attend, 0=pad
            # 扩展为 4D: [bsz, 1, seq_len, seq_len]
            expanded = attention_mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len).to(dtype)
            # 1 → 0.0 (attend), 0 → -inf (mask)
            return (1.0 - expanded) * torch.finfo(dtype).min

        # 无 padding: 全部 attend
        return torch.zeros(bsz, 1, seq_len, seq_len, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    #  DLLM: MDM-style noise (token masking)
    # ------------------------------------------------------------------
    def add_noise_to_tokens(self, input_ids: torch.LongTensor, t: torch.FloatTensor,
                            eps: float = None, pad_token_id: int = 0):
        """
        根据噪声水平 t 将 token 替换为 [MASK]。
        t=0 → 几乎不 mask,  t=1 → 全部 mask
        """
        batch_size, seq_len = input_ids.shape
        eps = eps if eps is not None else self.mask_epsilon
        p_mask = (1 - eps) * t + eps  # t=0→eps, t=1→1
        p_mask = p_mask.unsqueeze(-1).expand(batch_size, seq_len)
        corruption_mask = torch.rand(batch_size, seq_len, device=input_ids.device) < p_mask
        corruption_mask = corruption_mask & (input_ids != pad_token_id)  # 不 mask padding
        noisy_input_ids = torch.where(corruption_mask, self.mask_token_id, input_ids)
        return noisy_input_ids, corruption_mask, p_mask

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask=None, labels=None,
                corruption_mask=None, p_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None and corruption_mask is not None and p_mask is not None:
            # DLLM 加权交叉熵: loss[mask] / p_mask[mask]
            ce = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                reduction="none",
            ).view(labels.shape)
            loss = (ce[corruption_mask] / p_mask[corruption_mask]).sum() / labels.numel()

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)



if __name__ == "__main__":
    parser = HfArgumentParser((QwenDLLMTrainingArguments, TrainingArguments))
    args, trainer_args = parser.parse_args_into_dataclasses()

    # ----------------------------------------------------------------
    #  Tokenizer
    # ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.model_max_length = args.max_seq_len

    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 添加 [MASK] token (dLLM 必需)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    mask_token_id = tokenizer.mask_token_id

    # ----------------------------------------------------------------
    #  Model: 加载 Qwen2 预训练权重，转换为 dLLM
    # ----------------------------------------------------------------
    model = QwenForMaskedDiffusion.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    # 设置 dLLM 配置
    model.mask_token_id = mask_token_id
    model.mask_epsilon = args.mask_epsilon
    model.config.mask_token_id = mask_token_id
    model.config.mask_epsilon = args.mask_epsilon

    # 如果添加了新 token (如 [MASK])，需要 resize embedding
    model.resize_token_embeddings(len(tokenizer))

    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    print(f"mask_token_id: {mask_token_id}, mask_token: {tokenizer.mask_token}")

    # ----------------------------------------------------------------
    #  数据集 & Collator
    # ----------------------------------------------------------------
    dataset = JsonPretrainDataset(args.dataset_path, tokenizer=tokenizer)

    collator = DLLMDataCollator(
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        mask_epsilon=args.mask_epsilon,
    )

    # ----------------------------------------------------------------
    #  训练
    # ----------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    train_result = trainer.train()

    # 保存模型和 tokenizer
    trainer.save_model(trainer_args.output_dir)
    tokenizer.save_pretrained(trainer_args.output_dir)
