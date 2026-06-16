# microLLM — 从零理解大模型

一个**教学用**的微型语言模型项目：用最少的代码，从头实现一个能训练、能生成文本的字符级模型，借此搞懂大模型（LLM / Transformer）的核心原理。

> 目标不是做一个强大的模型，而是让每一个名词（token、embedding、attention、loss、梯度……）都能在代码里指出对应的那几行。

---

## 它在做一件什么事

本质极其简单：**给定前面的字符，预测下一个字符**。反复执行这个预测，就能一个字一个字地生成文本。

```
输入 "To be or not to b" → 模型预测下一个字符大概率是 "e"
```

训练数据是莎士比亚剧本（`tiny_shakespeare.txt`，约 110 万字符）。

---

## 数据流（一张图看懂整个项目）

```
文本文件
  │  read_file
  ▼
一长串字符 ──Tokenizer.encode──▶ 一长串数字(token) ──▶ 一维张量(data)
  │  train_val_split
  ▼
train / val ──get_batch──▶ 一批 (x, y)   # y 是 x 右移一位 = "每个位置的下一个字符"
  │  model(x, y)
  ▼
Model: Embedding 查表 → logits(每个位置对65个字符的打分) ──cross_entropy(与y比)──▶ loss
  │  loss.backward() + optimizer.step()   # 反向传播 + 更新参数
  ▼
重复几千步，loss 下降，模型学会规律
  │  model.generate()
  ▼
从一个起始字符出发，循环采样 → 生成新文本
```

---

## 项目结构

```
LearnLLM/
├── microLLM/
│   ├── tokenizer.py    # Tokenizer 类：建词表 + encode/decode（字符↔数字）
│   ├── data.py         # read_file / make_data_tensor / train_val_split / get_batch
│   ├── attention.py    # Head 类：单头自注意力（Q/K/V → 缩放点积 → 因果掩码 → softmax → @V）
│   ├── model.py        # Model：迷你 Transformer（token+position embedding → sa_head → lm_head）
│   ├── train.py        # 训练入口：装配 → 优化器 → 训练循环 → 生成对比
│   ├── test.py         # 临时测试脚本（验证形状、loss、生成）
│   ├── generate.py     # （预留）训练后加载模型生成文本的入口脚本
│   └── tiny_shakespeare.txt  # 训练数据
├── docs/
│   └── QA.md           # 学习过程中所有疑问的整理（强烈建议复习）
└── README.md
```

---

## 怎么运行

环境用 [uv](https://docs.astral.sh/uv/) 管理（Python 3.12 + PyTorch CPU 版）。

```bash
# 训练并在结束后生成一段文本
uv run python microLLM/train.py
```

训练时会看到 loss 从约 4.6 下降并稳定在约 2.4（这是 Bigram 的天花板）。

---

## 核心概念速查

| 名词 | 是什么 | 在代码里 |
|---|---|---|
| token | 模型处理的最小单位（这里=单个字符） | `Tokenizer.encode` |
| vocab_size | 不同字符的种类数（这里=65） | `Tokenizer.vocab_size` |
| 张量 tensor | 带类型和形状的多维数组 | 全程 |
| embedding | 把 token 编号变成向量的查找表 | `nn.Embedding` |
| logits | 模型输出的"下一个字符各候选的分数"(float) | `forward` 返回值 |
| loss | 预测和真实答案的差距（交叉熵） | `F.cross_entropy` |
| 梯度下降 | 靠 loss 指方向，一步步调整参数 | `loss.backward()` + `optimizer.step()` |
| block_size | 上下文长度（模型能看多远）→ **模型能力** | 超参数 |
| batch_size | 一步并行几条序列 → **训练稳定性/速度** | 超参数 |

详细的原理问答见 [docs/QA.md](docs/QA.md)。

---

## 进度与路线

- [x] **环境**：uv + PyTorch
- [x] **Tokenizer**：字符级 encode/decode
- [x] **数据管线**：读取、编码、train/val 切分、批次采样
- [x] **Bigram 模型**：Embedding → logits → loss → 训练 → 生成（loss ≈ 2.4）
- [x] **单头自注意力**：Q/K/V → 缩放点积 → 因果掩码 → softmax → 加权混合（attention.py）
- [x] **位置编码 + 迷你 Transformer**：token+position embedding → sa_head → lm_head（loss ≈ 2.3，文本现剧本结构）
- [ ] **Multi-Head Attention**：多个头并行看不同关系，再合并 ← 下一步
- [ ] **Feed-Forward（FFN）**：注意力后每个位置单独"消化"
- [ ] **残差连接 + LayerNorm**：稳定深层训练
- [ ] **堆叠多个 Block + 调大规模**：loss 可降到 ≈ 1.5，文本更连贯

---

## 当前进展：从 Bigram 到迷你 Transformer

- **Bigram** 只看前一个字符，loss 到 ≈2.4 就到头了。
- 加上**自注意力 + 位置编码**后，每个位置能"回看全部上文并按相关性加权利用"，loss 降到 ≈2.3，生成文本开始出现剧本结构和真单词碎片。
- 提升仍有限，因为目前是**单头、单层、block_size 仅 8**。后续加多头注意力、FFN、残差+LayerNorm、堆叠多层并调大规模，loss 可降到 ≈1.5，文本更连贯。

> 详细的逐题原理见 [docs/QA.md](docs/QA.md)，已覆盖 tokenization、张量、神经网络与梯度下降、预测机制、Q/K/V 自注意力全流程、迷你 Transformer 结构、以及调试经验。
