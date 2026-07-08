from attention import MultiHeadAttention
import torch.nn as nn
import torch

class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, num_heads, block_size)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x