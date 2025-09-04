import os
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from datasets import load_dataset
from dataclasses import dataclass

from hf_config import ModelConfig
from hf_model import MyGPTForCausalLM
from preprocess import process_pretrain

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
    def process_dataset(item):
        text = item['text']
        return process_pretrain(text, tokenizer)
    dataset = load_dataset("json", data_files=args.dataset_path, split='train')
    dataset = dataset.map(process_dataset, num_proc=(os.cpu_count() or 1), remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling( # 动态padding
        tokenizer=tokenizer,
        mlm=False,
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