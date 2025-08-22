# Data Pipeline

一个**完整、工业级 Hugging Face / PyTorch 训练数据流水线**，从原始数据到最终送入模型的 `Tensor`，并说明每一步产生的中间数据、作用，以及注意事项。


## **1️⃣ 原始数据加载阶段**

### 数据来源：

* JSON / CSV / 文本文件
* 数据库 / 自定义 Python Dataset
* Hugging Face Hub dataset

### 典型代码：

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.json")
# 输出：DatasetDict({'train': Dataset(...)})
```

### 中间数据：

* `dataset['train']` 是 `datasets.Dataset` 类型
* 每条样本是 **dict**，例如：`{"text": "这是第一条语料"}`
* 此时没有 token，只是原始字符串

> ✅ 注意：此阶段**不需要 tokenizer**，保留原始文本是好习惯

---

## **2️⃣ Tokenization 阶段**

### 什么时候做：

* 在模型训练前必须把文本转成 token IDs
* 对 NLP 来说，通常用 **Hugging Face tokenizer**

### 典型代码：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
max_length = 128

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True,
        padding="max_length",    # 或者保留到 longest
        max_length=max_length
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

### 中间数据：

* `tokenized_dataset['train']` 每条样本：

```python
{
    "text": "这是第一条语料",
    "input_ids": [101, 782, 102, 0, 0, 0, ...],  # 长度 = max_length
    "attention_mask": [1, 1, 1, 0, 0, 0, ...]
}
```

* `input_ids` → 模型输入
* `attention_mask` → 控制 padding token 的 attention
* 如果是 MLM 任务，会额外生成 `labels` 或 `loss_mask`

> 🔑 核心：这个阶段 **每条样本都是 dict**，长度可能固定，也可以动态 padding

---

## **3️⃣ DataCollator 阶段**

### 作用：

* 把 batch 样本 **从 list of dict 转成 tensor**
* 支持：

  * **动态 padding**（batch 内对齐最长序列）
  * **生成 labels / MLM mask**
  * **loss mask / ignore\_index**

### 典型代码：

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)
```

* 输入：batch of dict `[{"input_ids":..., "attention_mask":...}, ...]`
* 输出：

```python
{
    "input_ids": Tensor[batch, seq_len],
    "attention_mask": Tensor[batch, seq_len],
    "labels": Tensor[batch, seq_len]  # MLM mask applied
}
```

> 注意：collator 是 **DataLoader 层的最后一步**，在 sampler 之后，batch 划分好后才执行

---

## **4️⃣ Sampler / DataLoader 阶段**

### Sampler 作用：

* 控制 batch 的顺序
* 支持：

  * **随机打乱（shuffle）** → 训练阶段常用
  * **顺序读取** → 验证阶段
  * **按长度分组（LengthGroupedSampler）** → 减少 padding 浪费

### DataLoader 典型写法：

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_dataset['train'],
    batch_size=16,
    shuffle=True,                     # 使用随机 sampler
    collate_fn=data_collator          # collator 将 list[dict] → batch tensors
)
```

* **输入给 DataLoader**：dataset（每条样本是 dict）
* **输出给训练循环**：batch tensors（input\_ids, attention\_mask, labels 等）

---

## **5️⃣ 最终送入模型**

### Batch 结构：

```python
{
    "input_ids": Tensor[batch, seq_len],
    "attention_mask": Tensor[batch, seq_len],
    "labels": Tensor[batch, seq_len]  # 可选
}
```

* input\_ids → Transformer embedding 层
* attention\_mask → Transformer 自注意力 mask
* labels → loss 计算

### 示例训练循环：

```python
for batch in train_dataloader:
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch.get("labels", None)
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## **6️⃣ 数据流程总结（流水线视图）**

```
原始数据（文本 JSON/CSV） 
        │
        ▼
datasets.Dataset['train'] 每条样本 dict
        │ map(tokenize)
        ▼
tokenized_dataset 每条样本 dict
{
    "input_ids": [...],
    "attention_mask": [...],
    "text": ...
}
        │ DataLoader + Sampler (shuffle / length group)
        ▼
batch of samples (list of dict)
        │ collate_fn
        ▼
batch tensors
{
    "input_ids": Tensor[batch, seq_len],
    "attention_mask": Tensor[batch, seq_len],
    "labels": Tensor[batch, seq_len]
}
        │
        ▼
模型 forward → logits → loss
```

---

# **7️⃣ 业界主流规范**

1. **保持原始数据到最后一刻再 tokenize**

   * 避免硬编码 max\_length → 方便动态 padding
2. **Tokenizer map → DataCollator → DataLoader**

   * 保证 batch 统一长度，loss mask / attention mask 自动生成
3. **Sampler / shuffle**

   * 训练随机，验证顺序即可
4. **batch tensors**

   * 送入模型前最好转 device + dtype
5. **训练循环简洁**

   * batch tensors → forward → loss → backward → step

> 这是 Hugging Face + PyTorch / Transformers 的最主流写法
