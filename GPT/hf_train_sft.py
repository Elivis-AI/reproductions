from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from hf_model import MyGPTForCausalLM
from dataset import JsonSFTDataset

from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 在SFT这一步学会对话格式和<|im_start|><|im_end|>这两个对话控制符
CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system_message = messages[0]['content'] %}"
    "{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}"
    "{% else %}"
    "{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% set content = message['content'] %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% generation %}"    # for `DataCollatorForSeq2Seq` generate labels
    "{{ content + '<|im_end|>' + '\\n' }}"
    "{% endgeneration %}" # for `DataCollatorForSeq2Seq` generate labels
    "{% endif %}"
    "{% endfor %}"
)

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
    tokenizer.chat_template = CHAT_TEMPLATE
    model = MyGPTForCausalLM.from_pretrained(args.model_path)
    # print(f'参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M')

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
    trainer.save_model(trainer_args.output_dir)
    tokenizer.save_pretrained(trainer_args.output_dir)