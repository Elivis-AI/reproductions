from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from model import LLaVaConfig, MiniLLaVa
from processor import myLlavaProcessor
from dataset import JsonPretrainDataset

from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 和数据集格式对应好
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
    
    # model config
    llm_model: str
    clip_model: str
    patch_size: int
    image_token: str = "<image>" # 注意和数据集对应好
    max_seq_len: int = None


if __name__ == "__main__":
    parser = HfArgumentParser((MyTrainingArguments, TrainingArguments))
    args, trainer_args = parser.parse_args_into_dataclasses()
    # args: MyTrainingArguments = parser.parse_args_into_dataclasses()[0]

    # 初始化 processor 和 model
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    tokenizer.chat_template = CHAT_TEMPLATE
    image_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    model_config = LLaVaConfig(**vars(args))
    if args.image_token not in tokenizer.vocab: # 扩展词表，添加<image>
        tokenizer.add_special_tokens({"additional_special_tokens": [args.image_token]})
    model_config.image_token_id = tokenizer.convert_tokens_to_ids(args.image_token)
    processor = myLlavaProcessor(image_processor=image_processor, tokenizer=tokenizer, patch_size=args.patch_size, chat_template=CHAT_TEMPLATE)
    model = MiniLLaVa(model_config)

    
    # 数据准备
    dataset = JsonPretrainDataset(args.dataset_path, processor)
    
    if not hasattr(tokenizer, 'pad_token'):
        tokenizer.pad_token = tokenizer.eos_token
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
    processor.save_pretrained(trainer_args.output_dir)