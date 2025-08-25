import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Any, List, Dict


class PretrainDataset(Dataset):
    dataset: Any # 必须支持index访问
    tokenizer: AutoTokenizer
    
    def __init__(self, dataset: str, tokenizer: AutoTokenizer):
        raise NotImplementedError
    
    def extract_text(self, item: Any) -> str:
        raise NotImplementedError
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        text = self.extract_text(item)
        
        # tokenize
        input_ids: torch.Tensor = self.tokenizer.encode(
            text,
            return_tensors="pt",    
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        
        return {"input_ids": input_ids.squeeze(0)}


class SFTDataset(Dataset):
    dataset: Any # 必须支持index访问
    tokenizer: AutoTokenizer

    def __init__(self, dataset: str, tokenizer: AutoTokenizer):
        raise NotImplementedError
    
    def extract_messages(self, item: Any) -> List[Dict[str, str]]:
        """构建符合模板格式的对话
        {"role": "system", "content": "你是一个聊天机器人"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
        """
        # messages = []
        # for i, turn in enumerate(conversations):
        #     role = 'user' if i % 2 == 0 else 'assistant'
        #     messages.append({"role": role, "content": turn['content']})
        raise NotImplementedError
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        messages = self.extract_messages(item)
        
        # 构建对话提示
        output = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=True,
            padding=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        input_ids = output.input_ids.squeeze()
        labels = input_ids.masked_fill(torch.tensor(output.assistant_masks) == False, self.tokenizer.pad_token_id)

        return {"input_ids": input_ids, "labels": labels}
    

class ZhiHuPretrainDataset(PretrainDataset):
    def __init__(self, dataset: str, tokenizer: AutoTokenizer):
        self.dataset = load_dataset(dataset, split='train')
        self.tokenizer = tokenizer

    def extract_text(self, item):
        text = item['INSTRUCTION']  + '\n\n' + item['RESPONSE']
        text = text + self.tokenizer.eos_token  # 添加<|endoftext|>
        return text


class MiniMindPretrainDataset(PretrainDataset):
    def __init__(self, dataset: str, tokenizer: AutoTokenizer):
        self.dataset = load_dataset("json", data_files=dataset, split='train')
        self.tokenizer = tokenizer

    def extract_text(self, item):
        return item['text']


class MiniMindSFTDataset(SFTDataset):
    def __init__(self, dataset: str, tokenizer: AutoTokenizer):
        self.dataset = load_dataset("json", data_files=dataset, split='train')
        self.tokenizer = tokenizer

    def extract_messages(self, item):
        return item['conversations']