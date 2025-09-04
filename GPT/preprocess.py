import torch
from transformers import AutoTokenizer
from typing import  List, Dict

def process_pretrain(text: str, tokenizer: AutoTokenizer):
    # tokenize
    output = tokenizer(
        text,
        return_tensors="pt",
        return_token_type_ids=False,
        padding=False,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    
    return {"input_ids": output.input.squeeze()}


def process_sft(messages: List[Dict[str, str]], tokenizer: AutoTokenizer):
    """messages: 符合模板格式的对话
    e.g.
    [
        {"role": "system", "content": "你是一个聊天机器人"},
        {"role": "user", "content": '请介绍一下你自己。'},
        {"role": "assistant", "content": '您好，我是一个聊天机器人，请问有什么可以帮您的吗？'},
    ]
    """
    # 构建对话提示
    assert hasattr(tokenizer, 'chat_template'), "tokenizer缺少chat_template"
    output = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        padding=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    input_ids = output.input_ids.squeeze()
    labels = input_ids.masked_fill(torch.tensor(output.assistant_masks) == False, -100) # -100是default ignore_index

    return {"input_ids": input_ids, "labels": labels}


    
if __name__ == '__main__':
    # example usage
    import os
    from datasets import load_dataset, load_from_disk

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    dataset = load_dataset("json", data_files='pretrain_hq_v7.jsonl', split='train')

    def process_dataset(item):
        text = item['text']
        return process_pretrain(text, tokenizer)

    dataset = dataset.map(process_dataset, num_proc=(os.cpu_count() or 1), remove_columns=dataset.column_names)
    
    # 持久化tokenized dataset
    dataset.save_to_disk("tokenized_pretrain_hq_v7")
    dataset = load_from_disk("tokenized_pretrain_hq_v7")
