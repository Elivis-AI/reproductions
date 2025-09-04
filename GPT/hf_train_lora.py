import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from peft import get_peft_model, LoraConfig, TaskType

from hf_model import MyGPTForCausalLM
from dataset import JsonSFTDataset

from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MyTrainingArguments:
    # data params
    dataset_path: str
    model_path: str
    max_seq_len: int

    # lora
    rank: int = 8
    alpha: int = 8
    dropout: float = 0.1
    merge_weights: bool = True


if __name__ == "__main__":
    parser = HfArgumentParser((MyTrainingArguments, TrainingArguments))
    args, trainer_args = parser.parse_args_into_dataclasses()
    # args: MyTrainingArguments = parser.parse_args_into_dataclasses()[0]

    # 初始化 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MyGPTForCausalLM.from_pretrained(args.model_path)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=args.rank, 
        lora_alpha=args.alpha, 
        lora_dropout=args.dropout,
        target_modules=["w_q", "w_k", "w_v", "w_out"], # don't apply to ffn layers
    )

    # 加载lora参数
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 数据准备
    dataset = JsonSFTDataset(args.dataset_path, tokenizer=tokenizer)
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
    if args.merge_weights: # 合并lora权重
        model = model.merge_and_unload()
    model.save_pretrained(trainer_args.output_dir) # trainer.save_model保存的只是lora权重
    tokenizer.save_pretrained(trainer_args.output_dir)