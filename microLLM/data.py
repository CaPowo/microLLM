#这个模块负责处理所有有关数据的部分，包括 文件读取，train/val（训练验证）划分，get_batch

import torch
import random

from tokenizer import BaseTokenizer

# 1) 读入整个文本文件（一长串字符）
def read_file(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# 用tokenizer整个文本编码成一维的long张量
def make_data_tensor(text: str,tokenizer: BaseTokenizer)-> torch.Tensor:
    #使用 torch.tensor 函数构造张量 用dtype 明确类型
    token = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return token

def train_val_split(data: torch.Tensor, ratio: float = 0.9):
    """按比例切成 (train_data, val_data) 两个张量。"""
    lenth = len(data)
    split_idx = int(ratio*lenth)

    return data[:split_idx],data[split_idx:]

def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    """
    随机采一个 batch。
    返回 (x, y)，形状都是 (batch_size, block_size)，
    batch_size 就是把材料分成几段 block_size 就是一次看几个
    其中 y 是 x 整体右移一位（每个位置的"下一个字符"就是标签）。
    """
    random_index = random.sample(range(len(data) - block_size), batch_size)

    x = torch.stack([data[i:i+block_size] for i in random_index])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_index])

    return x,y
