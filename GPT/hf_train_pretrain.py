import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from hf_config import ModelConfig
from hf_model import MyGPTForCausalLM

import warnings
warnings.filterwarnings('ignore')

@dataclass
class MyTrainingArguments:
    # data params
    dataset_path: str
    tokenizer: str

    # model params
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 6400
    max_seq_len: int = 8192
    padding_idx: int = None
    rope_theta: int = 1e6
    dropout: float = 0.0
    flash_attn: bool = True



class ZhiHuPretrainDataset(Dataset):
    def __init__(self, dataset: str, tokenizer: AutoTokenizer):
        self.dataset = load_dataset(dataset, split='train')
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        text = item['INSTRUCTION']  + '\n\n' + item['RESPONSE']
        text = text + self.tokenizer.eos_token  # 添加<|endoftext|>
        
        # tokenize
        input_ids: torch.Tensor = self.tokenizer.encode(
            text,
            return_tensors="pt",    
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        
        return {"input_ids": input_ids.squeeze(0)}

class MyDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch["labels"] = batch["input_ids"].detach()
        return batch


if __name__ == "__main__":
    parser = HfArgumentParser((MyTrainingArguments, TrainingArguments))
    args, trainer_args = parser.parse_args_into_dataclasses()
    # args: MyTrainingArguments = parser.parse_args_into_dataclasses()[0]

    # 初始化 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer.model_max_length = args.max_seq_len
    args.padding_idx = tokenizer.pad_token_id
    model_config = ModelConfig(**vars(args))
    model = MyGPTForCausalLM(model_config)
    # print(f'参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M')

    # 数据准备
    dataset = ZhiHuPretrainDataset(dataset=args.dataset_path, tokenizer=tokenizer)
    collator = MyDataCollatorWithPadding( # 动态padding
        tokenizer=tokenizer,
        padding=True,
        max_length=tokenizer.model_max_length
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # print("开始训练")
    train_result = trainer.train()


    # print(f"训练完成，模型已保存至{trainer_args.output_dir}")
    trainer.save_model(trainer_args.output_dir)
    tokenizer.save_pretrained(trainer_args.output_dir)