import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.modeling_outputs import MaskedLMOutput

from dllm_utils import DllmMixin
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

class QwenForMaskedDiffusion(Qwen2PreTrainedModel, DllmMixin):
    """
    基于 Qwen2 架构的扩散语言模型 (dLLM)。
    """
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 双向注意力
        for layer in self.model.layers:
            layer.self_attn.is_causal = False
        self.model._update_causal_mask = self._bidirectional_mask
        
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
    #  Forward: 输入与常规 LLM 一致, 噪声注入自动完成
    # ------------------------------------------------------------------
    def forward(
            self, 
            input_ids: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            labels: Optional[torch.Tensor] = None, 
            **kwargs
        ):
        
        corruption_mask, p_mask = None, None
        if labels is not None and self.training:
            input_ids, corruption_mask, p_mask = self.add_noise_to_tokens(
                input_ids, 
                eps=getattr(self.config, "mask_epsilon", 0.001),
                pad_token_id=getattr(self.config, "pad_token_id", 0),
                mask_token_id=getattr(self.config, "mask_token_id", self.config.vocab_size - 1),
            )

        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        logits: torch.Tensor = self.lm_head(hidden_states).float()

        loss = None
        if all([x is not None for x in (corruption_mask, p_mask, labels)]):
            loss = self.dllm_loss(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1), 
                corruption_mask, 
                p_mask
            )

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)



if __name__ == "__main__":
    parser = HfArgumentParser((QwenDLLMTrainingArguments, TrainingArguments))
    args, trainer_args = parser.parse_args_into_dataclasses()

    # ----------------------------------------------------------------
    #  断点检测: 支持 --resume_from_checkpoint auto
    # ----------------------------------------------------------------
    resume_checkpoint = trainer_args.resume_from_checkpoint
    if resume_checkpoint == "auto":
        output_dir = trainer_args.output_dir
        if os.path.isdir(output_dir):
            ckpts = [
                os.path.join(output_dir, d)
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
            ]
            resume_checkpoint = max(ckpts, key=os.path.getmtime) if ckpts else None
        else:
            resume_checkpoint = None
        if resume_checkpoint:
            print(f"自动检测到 checkpoint: {resume_checkpoint}")
        else:
            print("未检测到已有 checkpoint，从头开始训练")
    elif resume_checkpoint and resume_checkpoint is not True and not os.path.isdir(str(resume_checkpoint)):
        print(f"警告: 指定的 checkpoint 路径不存在: {resume_checkpoint}，将从头开始训练")
        resume_checkpoint = None

    # 确定模型加载路径: 有 checkpoint 则从 checkpoint 加载, 否则从原始预训练模型
    model_path = resume_checkpoint if resume_checkpoint and resume_checkpoint is not True else args.model_name_or_path

    # ----------------------------------------------------------------
    #  Tokenizer
    # ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
        model_path,
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

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不使用 HF 的 MLM, 噪声注入在 model.forward() 中自动完成
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

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 保存模型和 tokenizer
    trainer.save_model(trainer_args.output_dir)
    tokenizer.save_pretrained(trainer_args.output_dir)
