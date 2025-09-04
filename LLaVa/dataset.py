import os
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset

from processor import myLlavaProcessor
from typing import Any, Dict


class VLMPretrainDataset(Dataset):
    data: Any
    processor: myLlavaProcessor

    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def build_output(self, messages, images):
        output = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=False,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            truncation=True,
        )
        labels = output.input_ids.masked_fill(torch.tensor(output.assistant_masks) == False, -100) # -100是default ignore_index

        text = self.processor.apply_chat_template(messages, truncation=True)
        output = self.processor(images=images, text=text)
        labels = F.pad(labels, (output.input_ids.size(1) - labels.size(1), 0), value=-100)

        return {
            "input_ids": output.input_ids.squeeze(),
            "pixel_values": output.pixel_values.squeeze(),
            "attention_mask": output.attention_mask.squeeze(),
            "labels": labels.squeeze(),
        }
    
    def __len__(self):
        return len(self.data)

    def preprocess(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, idx):
        item = self.data[idx]
        output = self.preprocess(item)
        assert output.keys() == {'messages', 'images'}
        return self.build_output(**output)
    

class ExamplePretrainDataset(VLMPretrainDataset):
    def __init__(self, data_path, processor):
        data = load_dataset("json", data_files=data_path, split='train')
        self.data_root = os.path.dirname(data_path)
        super().__init__(data, processor)
    
    def preprocess(self, item: Dict[str, Any]) -> Dict[str, Any]:
        messages = item['conversations']
        img_path = os.path.join(self.data_root, 'pretrain_images', item['image'])
        image = Image.open(img_path).convert('RGB')
        return {'messages': messages, 'images': image}