import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, n_embd: int,head_size: int, block_size: int):
        super().__init__()

        # 三个学习矩阵 Wq/Wk/Wv（就是 nn.Linear，bias=False 表示纯矩阵乘法）
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 下三角掩码：它不是参数(不学习)，但要随模型走，用 register_buffer 登记
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        # ① 算 Q/K/V（每个 (B, T, head_size)）
        q = self.query(x)

        k = self.key(x)

        v = self.value(x)

        # ② 算亲和度 Q·K：要把 k 的最后两维转置才能相乘
        wei = q @ k.transpose(-2, -1)

        # ③ 缩放（除以 √head_size，见下方说明）
        wei = wei * self.head_size ** -0.5

        # ④ mask 未来：把上三角设 -inf（用 self.tril[:T, :T]）
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # ⑤ softmax 成权重
        wei = F.softmax(wei, dim= -1)       
        # ⑥ 加权混合 V
        out = wei @ v       
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size)
            for _ in range(num_heads)
        ]) #起循环 建立多注意力头
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
