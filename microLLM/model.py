# model的作用 是基于已有的参数(一堆可学习的数字) 以及一套计算规则，来算出下一个词的概率
import torch
import torch.nn as nn
from  torch.nn import functional as F
from attention import MultiHeadAttention

class FeedForward(nn.Module):  #构造FFN
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class Model(nn.Module):
    # 制作空白的“分数表”
    def __init__(self,vocab_size:int,n_embd: int, block_size: int, num_heads: int):
        super().__init__()
        assert n_embd % num_heads == 0

        self.block_size = block_size

        # token：编号 → n_embd 维含义向量
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        
        # 位置：位置 → n_embd 维位置向量
        self.position_embedding_table = nn.Embedding(block_size,n_embd)

        # 注意力头（这里引入多注意力头 num_heads= 决定头的数量）
        self.sa_head = MultiHeadAttention(n_embd, num_heads=num_heads, block_size=block_size)        # 输出层：n_embd → vocab_size 个分数
        
        self.ffn = FeedForward(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)


    # 预测
    def forward(self, idx:torch.Tensor, targets:torch.Tensor=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"输入序列长度 T={T} 超过模型最大 block_size={self.block_size}，请同步修改训练采样长度和 Model(...) 里的 block_size。")

        #token 含义向量
        tok_emb = self.token_embedding_table(idx)
        #位置向量
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        #相加
        x = tok_emb + pos_emb

        x = self.sa_head(x)

        x = self.ffn(x)

        logits = self.lm_head(x)

        #这是个三维元组 分别对应 B T C 
        #其中 B 是分块数量 T是每个分块的大小 C是总共的分数(也就是vocab_size)
        #生成阶段没有答案不算loss
        if targets is None:
            loss = None
        else: 
            #训练时 需要计算差距
            #首先要做格式转换，需要reshape
            B1,T1,C1 = logits.shape
            B2,T2 = targets.shape

            #进行变形，把三维(第几行，第几列，多少分)转化为二维度(第几个，多少分)
            #随后进行差距运算
            loss = F.cross_entropy(logits.view(B1*T1,C1),targets.view(B2*T2))
        '''
        随机初始化时，模型对 65 个字符是"瞎猜"，理论 loss ≈ ln(65) ≈ 4.17。
        你跑出来如果在 4.x 这个量级，就说明一切正常——模型确实在"均匀瞎猜"，还没学习。
        等训练后你会看到它往下掉。
        '''
        return logits,loss
    
    @torch.no_grad() #生成时不需要训练，直接关闭求导
    def generate(self, idx:torch.Tensor, max_new_tokens: int):
        #idx:(B,T) 分块数量，每个分块的大小
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]   # 关键：只保留最后 block_size 个

            #1. 向前预测，得到 B组语料 每组语料T个token 一共B*T*C的预测结果
            logits, loss = self(idx_cond)

            #2. 只取最后一个位置预测
            logits = logits[:,-1,:]

            #3. softmax 把分数变成概率
            probs = F.softmax(logits,dim=-1)

            #4. 按概率抽取一个
            next_id = torch.multinomial(probs,1)

            #5. 把next_id拼接到idx末尾
            idx = torch.cat((idx,next_id), dim=1)

        return idx
