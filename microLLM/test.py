#用于中途测试
import torch

from tokenizer import Tokenizer
from data import read_file, make_data_tensor, train_val_split, get_batch
from model import Model

text = read_file("C:\\Users\\15589\\Desktop\\LearnLLM\\microLLM\\tiny_shakespeare.txt")
tok = Tokenizer(text)
data = make_data_tensor(text, tok)
train_data, val_data = train_val_split(data)

xb, yb = get_batch(train_data, block_size=8, batch_size=4)

model = Model(tok.vocab_size)
logits, loss = model(xb, yb)
print(data)
print(logits.shape)   # torch.Size([4, 8, 65])
print(loss)           # 期望 4.x 左右

start = torch.zeros((1, 1), dtype=torch.long)
out = model.generate(start, max_new_tokens=300)
print(tok.decode(out[0].tolist()))
