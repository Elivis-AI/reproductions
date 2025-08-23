from datasets import load_dataset
from transformers import (
    HfArgumentParser, 
    AutoTokenizer,
    PreTrainedTokenizerFast
)
from tokenizers import (
    pre_tokenizers,
    models,
    processors,
    decoders,
    trainers,
    Tokenizer,
)

from dataclasses import dataclass

import random
random.seed(42)

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
    "{{ content + '<|im_end|>' + '\\n' }}"
    "{% endif %}"
    "{% endfor %}"
)


@dataclass
class TokeizerArguments:
    dataset_path: str
    max_len: str = 512
    vocab_size: int = 6400
    save_path: str = './weights'
    template: str = None


def load_data(dataset_path: str):
    texts = load_dataset("json", data_files=dataset_path, split='train')
    for item in texts:
        yield item['text']


def train_tokenizer(args: TokeizerArguments):
    # 配置tokenizer
    # HF tokenizers 的主要pipeline设计如下：
    # 1. Model：定义 tokenization 算法（BPE/WordPiece/Unigram/WordLevel）
    # 2. PreTokenizer：如何切分输入文本（按字节、空格、正则等）
    # 3. PostProcessor：如何在输出序列上加修饰（特殊符号、offset 处理等）
    # 4. Decoder：如何从 token 序列恢复回原始文本
    # 业界最常见的组合是：
    # GPT 系列 → BPE + ByteLevel PreTokenizer + ByteLevel Decoder + ByteLevel PostProcessor
    # BERT 系列 → WordPiece + Whitespace PreTokenizer + TemplateProcessing (加CLS/SEP)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False) # by default

    # 定义特殊token

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"], # OpenAI风格的分隔符
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据
    iterator = load_data(args.dataset_path)

    # 训练tokenizer
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 转换成PreTrainedTokenizerFast类
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    pretrained_tokenizer.pad_token = "<|endoftext|>"
    pretrained_tokenizer.unk_token = "<|endoftext|>"
    pretrained_tokenizer.bos_token = "<|im_start|>"
    pretrained_tokenizer.eos_token = "<|im_end|>"

    pretrained_tokenizer.chat_template = CHAT_TEMPLATE
    pretrained_tokenizer.save_pretrained(args.save_path)
    print("Tokenizer training completed and saved.")


def train_tokenizer_from_template(args: TokeizerArguments):
    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.template, trust_remote_code=True)

    # 读取文本数据
    iterator = load_data(args.dataset_path)

    # 继承pretrained tokenizer的配置
    tokenizer.train_new_from_iterator(iterator, vocab_size=args.vocab_size)

    # 保存tokenizer
    tokenizer.save_pretrained(args.save_path)
    print("Tokenizer training completed and saved.")


def eval_tokenizer(args: TokeizerArguments):
    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.save_path)

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)


if __name__ == '__main__':
    parser = HfArgumentParser(TokeizerArguments)
    args = parser.parse_args_into_dataclasses()[0] # always return tuple

    if args.template:
        train_tokenizer_from_template(args)
    else:
        train_tokenizer(args)
    
    eval_tokenizer(args)