# Ref: https://github.com/jingyaogong/minimind/blob/master/eval_model.py
import argparse
import torch
from transformers import AutoTokenizer, TextStreamer
from hf_model import MyGPTForCausalLM

import warnings
warnings.filterwarnings('ignore')

# 设置可复现的随机种子
def setup_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--model', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--history_cnt', default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    if args.seed is not None:
        setup_seed(args.seed)
    
    model = MyGPTForCausalLM.from_pretrained(args.model).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = []
    for idx, prompt in enumerate(iter(lambda: input('>>> '), '')):

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(args.device)

        print('>>> ', end='')
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature
        )

        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        print('\n\n')