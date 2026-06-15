import torch
from attention import Head

torch.manual_seed(1337)
B, T, n_embd, head_size = 1, 15, 8, 8
x = torch.randn(B, T, n_embd)

head = Head(n_embd, head_size, block_size=T)
out = head(x)
print(out.shape)        # 期望 (1, 15, 8)