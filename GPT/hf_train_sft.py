from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from hf_model import MyGPTForCausalLM
from dataset import MiniMindSFTDataset

import warnings
warnings.filterwarnings('ignore')

@dataclass
class MyTrainingArguments:
    # data params
    dataset_path: str
    model_path: str
    max_seq_len: int


if __name__ == "__main__":
    parser = HfArgumentParser((MyTrainingArguments, TrainingArguments))
    args, trainer_args = parser.parse_args_into_dataclasses()
    # args: MyTrainingArguments = parser.parse_args_into_dataclasses()[0]

    # 初始化 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MyGPTForCausalLM.from_pretrained(args.model_path)
    # print(f'参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M')

    # 数据准备
    dataset = MiniMindSFTDataset(dataset=args.dataset_path, tokenizer=tokenizer)
    collator = DataCollatorForSeq2Seq( # 动态padding
        tokenizer=tokenizer,
        padding=True,
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