# 负责训练数据减小loss
# 核心思路：取一批数据 → 前向算 loss → 反向算梯度 → 朝下坡挪一步 → 重复几千次

import torch
from tokenizer import Tokenizer
from data import read_file, make_data_tensor, train_val_split, get_batch
from model import Model

#1. 加载训练数据
text = read_file("C:\\Users\\15589\\Desktop\\LearnLLM\\microLLM\\tiny_shakespeare.txt")
tok = Tokenizer(text) #数据token化
data = make_data_tensor(text,tok) #构造一维张量
train, val = train_val_split(data) #分类 训练和验证集

#2. 超参数
batch_size = 32  #每一批的序列数
block_size = 64  #每一个序列里面的token数量
n_embd = 64
n_layer = 4
num_heads = 4
dropout = 0.2
learning_rate = 3e-4    #学习率
num_steps = 20000   #迭代次数
eval_interval = 500
eval_iters = 50
checkpoint_path = "microLLM/best_model.pt"


model = Model(n_embd=n_embd,block_size=block_size,vocab_size=tok.vocab_size,num_heads=num_heads, n_layer=n_layer, dropout=dropout)  #创建空白表格

#3. 优化器
"""
之前说"optimizer.step() 让每个旋钮挪一小步"。这个 optimizer 就是专门负责"根据梯度更新权重"的工具。我们用 AdamW——梯度下降的一个聪明升级版，自动帮你调好每个参数的步子大小，是现在的标配。
创建它时要告诉它两件事：① 管哪些参数（model.parameters()，PyTorch 自动收集的全部旋钮）；② 学习率（learning_rate）——每步挪多大，太大会"步子迈太大扯着走过头"，太小则学得慢。
骨
"""
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, data_src in [("train", train), ("val", val)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data_src, block_size, batch_size)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

best_val_loss = float("inf")

#4. 循环训练
# 训练就是步骤循环： 取一批数据 → 前向算 loss → 反向算梯度 → 朝下坡挪一步 → 重复
for i in range(num_steps):
    #采集batch
    xb ,yb  = get_batch(train, block_size, batch_size) 
    #前向，计算loss
    """
    model(xb, yb)
    ↓  自动转发，等价于
    model.forward(xb, yb)
    ↓  对应你的定义
    forward(self, idx=xb, targets=yb)
    """
    logits, loss = model(xb,yb)
    #反向计算梯度

    #清空上一轮的旧梯度
    optimizer.zero_grad(set_to_none=True)
    #计算梯度
    loss.backward()
    #按照梯度更新参数
    optimizer.step()

    '''
    train 降，val 也降：真进步
    train 降，val 不降：开始过拟合
    '''

    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": tok.vocab_size,
                "n_embd": n_embd,
                "block_size": block_size,
                "num_heads": num_heads,
                "n_layer": n_layer,
                "dropout": dropout,
                "best_val_loss": best_val_loss,
                "step": i,
            }, checkpoint_path)
            print(f"  saved best checkpoint: val loss {best_val_loss:.4f}")


start = torch.zeros((1, 1), dtype=torch.long)
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"loaded best checkpoint from step {checkpoint['step']}: val loss {checkpoint['best_val_loss']:.4f}")
out = model.generate(start, max_new_tokens=300, temperature=0.8)
print(tok.decode(out[0].tolist()))
