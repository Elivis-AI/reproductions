from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from model import MiniLLaVa
from processor import myLlavaProcessor
from dataset import JsonSFTDataset

from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MyTrainingArguments:
    # data params
    dataset_path: str
    model_path: str


if __name__ == "__main__":
    parser = HfArgumentParser((MyTrainingArguments, TrainingArguments))
    args, trainer_args = parser.parse_args_into_dataclasses()
    # args: MyTrainingArguments = parser.parse_args_into_dataclasses()[0]

    # 加载 processor 和 model
    processor = myLlavaProcessor.from_pretrained(args.model_path)
    model = MiniLLaVa.from_pretrained(args.model_path)

    # 数据准备
    dataset = JsonSFTDataset(args.dataset_path, processor)
    collator = DataCollatorForSeq2Seq( # 动态padding
        tokenizer=processor.tokenizer,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
        data_collator=collator,
    )

    # print("开始训练")
    train_result = trainer.train()


    # print(f"训练完成，模型已保存至{trainer_args.output_dir}")
    trainer.save_model(trainer_args.output_dir)
    processor.save_pretrained(trainer_args.output_dir)