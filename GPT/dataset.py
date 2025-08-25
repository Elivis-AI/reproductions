import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Any


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


class ZhiHuPretrainDataset(PretrainDataset):
    def __init__(self, dataset: str, tokenizer: AutoTokenizer):
        self.dataset = load_dataset(dataset, split='train')
        self.tokenizer = tokenizer

    def extrace_text(self, item):
        text = item['INSTRUCTION']  + '\n\n' + item['RESPONSE']
        text = text + self.tokenizer.eos_token  # 添加<|endoftext|>
        return text


class MiniMindPretrainDataset(PretrainDataset):
    def __init__(self, dataset: str, tokenizer: AutoTokenizer):
        self.dataset = load_dataset("json", data_files=dataset, split='train')
        self.tokenizer = tokenizer

    def extrace_text(self, item):
        return item['text']


