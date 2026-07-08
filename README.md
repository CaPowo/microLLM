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
Model: token+position embedding → 多层 Transformer Block → logits ──cross_entropy(与y比)──▶ loss
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
│   ├── attention.py    # Head / MultiHeadAttention：Q/K/V → 因果自注意力 → 多头拼接
│   ├── block.py        # Transformer Block：LayerNorm + 残差 + Multi-Head Attention + FFN
│   ├── model.py        # Model：embedding → 多层 Block → final LayerNorm → lm_head
│   ├── train.py        # 训练入口：训练循环 + train/val loss + best checkpoint
│   ├── test.py         # 临时测试脚本（验证形状、loss、生成）
│   ├── generate.py     # 加载 best_model.pt 并生成文本的入口脚本
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

# 训练后，加载 best checkpoint 单独生成文本
uv run python microLLM/generate.py --max-new-tokens 500 --temperature 0.8
```

训练初始 loss 通常在 4.x 左右。当前的多层 Transformer Block 版本，验证集 loss 已观察到约 1.6 左右，生成文本已经能学到剧本格式、换行、角色名和局部英文结构。

---

## 使用中文训练数据

模型主体不区分英文或中文，只要训练文本是 UTF-8 的 `.txt` 文件即可。把中文语料放到项目里，例如：

```text
microLLM/chinese.txt
```

默认仍可用字符级 tokenizer 训练：

```bash
uv run python microLLM/train.py --data microLLM/chinese.txt --checkpoint microLLM/chinese_best_model.pt --start "从前"
```

如果要使用更接近真实大模型的 BPE / subword tokenizer，加上 `--tokenizer bpe`：

```bash
uv run python microLLM/train.py --data microLLM/chinese.txt --checkpoint microLLM/chinese_bpe_model.pt --tokenizer bpe --bpe-merges 500 --start "从前"
```

对大语料，BPE 默认只使用前 `1,000,000` 个字符学习 merge 规则，但词表仍覆盖完整训练文本。可以手动调整：

```bash
# 用前 50 万字符学 BPE，通常启动更快
uv run python microLLM/train.py --data materials/AMC_scientific_articles_500K.txt --checkpoint microLLM/chinese_bpe_model.pt --tokenizer bpe --bpe-merges 300 --bpe-train-chars 500000

# 强制用全文学 BPE，不建议在很大语料上使用教学版 Python BPE
uv run python microLLM/train.py --data materials/AMC_scientific_articles_500K.txt --checkpoint microLLM/chinese_bpe_model.pt --tokenizer bpe --bpe-merges 500 --bpe-train-chars 0
```

CPU 训练阶段可以指定 PyTorch 线程数：

```bash
uv run python microLLM/train.py --data microLLM/chinese.txt --checkpoint microLLM/chinese_bpe_model.pt --tokenizer bpe --bpe-merges 500 --num-threads 8
```

`--num-threads 0` 或不写时，会自动使用系统能看到的逻辑核心数。注意：这个参数主要影响 PyTorch 的矩阵计算训练阶段；当前教学版 BPE 学 merge 规则仍然是 Python 单线程实现，所以大语料建议调小 `--bpe-train-chars` 和 `--bpe-merges`。

训练完成后生成：

```bash
uv run python microLLM/generate.py --checkpoint microLLM/chinese_bpe_model.pt --start "从前" --max-new-tokens 300 --temperature 0.8
```

字符级会把文本拆成单个字；BPE 会先从单字开始，再把高频相邻片段合并成更大的 token。比如训练后可能出现：

```text
字符级：人 / 工 / 智 / 能
BPE：人工 / 智能
```

BPE 的意义是让模型更容易稳定地产生常见词语或短语，但最终句子质量仍然取决于数据量、模型大小、训练时间和采样策略。

---

## 核心概念速查

| 名词 | 是什么 | 在代码里 |
|---|---|---|
| token | 模型处理的最小单位，字符级时是单个字符，BPE 时可以是常见字词片段 | `Tokenizer.encode` |
| vocab_size | token 种类数，字符级是不同字符数，BPE 是字符+合并片段数 | `Tokenizer.vocab_size` |
| 张量 tensor | 带类型和形状的多维数组 | 全程 |
| embedding | 把 token 编号变成向量的查找表 | `nn.Embedding` |
| logits | 模型输出的"下一个字符各候选的分数"(float) | `forward` 返回值 |
| loss | 预测和真实答案的差距（交叉熵） | `F.cross_entropy` |
| 梯度下降 | 靠 loss 指方向，一步步调整参数 | `loss.backward()` + `optimizer.step()` |
| block_size | 上下文长度（模型能看多远）→ **模型能力** | 超参数 |
| batch_size | 一步并行几条序列 → **训练稳定性/速度** | 超参数 |
| n_embd | 每个 token 在模型内部的向量宽度 | `Model(..., n_embd=32)` |
| attention | 让每个位置按权重读取上文信息 | `Head` / `MultiHeadAttention` |
| FFN | 注意力之后，对每个位置的向量单独加工 | `FeedForward` |
| residual | 把模块输出加回原输入，保留旧信息并稳定训练 | `Block.forward` |
| LayerNorm | 归一化每个 token 的内部向量，让深层训练更稳 | `nn.LayerNorm` |
| dropout | 训练时随机遮掉部分通道，降低过拟合 | `nn.Dropout` |
| checkpoint | 保存验证集 loss 最好的模型参数 | `best_model.pt` |

详细的原理问答见 [docs/QA.md](docs/QA.md)。

---

## 进度与路线

- [x] **环境**：uv + PyTorch
- [x] **Tokenizer**：字符级 encode/decode
- [x] **数据管线**：读取、编码、train/val 切分、批次采样
- [x] **Bigram 模型**：Embedding → logits → loss → 训练 → 生成（loss ≈ 2.4）
- [x] **单头自注意力**：Q/K/V → 缩放点积 → 因果掩码 → softmax → 加权混合（attention.py）
- [x] **位置编码 + 迷你 Transformer**：token+position embedding → sa_head → lm_head（loss ≈ 2.3，文本现剧本结构）
- [x] **Multi-Head Attention**：多个头并行看不同关系，再拼接 + projection 融合
- [x] **Feed-Forward（FFN）**：注意力后每个位置单独"消化"（loss 已观察到 ≈1.8）
- [x] **残差连接 + LayerNorm**：稳定深层训练
- [x] **堆叠多个 Block + 调大规模**：loss 可降到 ≈1.5~1.6，文本更连贯
- [x] **train/val loss + best checkpoint**：用验证集挑选最好模型，而不是盲用最后一步
- [x] **Dropout**：降低训练集记忆过强带来的过拟合
- [x] **正式生成脚本**：`generate.py` 加载 `best_model.pt` 生成文本
- [x] **可切换训练数据**：`train.py --data ...` 支持中文 UTF-8 文本
- [x] **BPE / subword tokenizer**：`--tokenizer bpe` 把高频相邻字符合并成常见片段
- [ ] **采样增强**：top-k / top-p，让生成更稳
- [ ] **工程化配置**：统一 config、设备选择（CPU/GPU）、更完整的 checkpoint 元信息

---

## 当前进展：Transformer 主体已完成

这个项目现在已经具备一个**字符级 decoder-only GPT 风格 Transformer** 的主体结构：

- token embedding + position embedding
- causal self-attention（因果掩码，不偷看未来）
- multi-head attention + output projection
- feed-forward network（FFN）
- residual connection + LayerNorm
- 多层 Transformer Block 堆叠
- final LayerNorm + lm_head
- 自回归 generate
- train/val loss 评估、best val loss checkpoint、dropout、temperature 采样

也就是说，从学习 Transformer 主干原理的角度看，主体已经完成。接下来不是继续"补 Transformer 核心积木"，而是进入**生成质量优化**和**训练脚本工程化**：比如 top-k/top-p 采样、学习率调度、GPU 支持、BPE tokenizer、更大数据和更大的模型。

> 详细的逐题原理见 [docs/QA.md](docs/QA.md)，已覆盖 tokenization、张量、神经网络与梯度下降、预测机制、Q/K/V 自注意力全流程、多头注意力、FFN、迷你 Transformer 结构、以及调试经验。
