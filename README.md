##这个项目是来做什么的？

这是一个LLM学习项目，用来理解基础的LLM原理，简单的基于pytorch实现一个微型模型

#项目结构

'''
microLLM/
├── config.py      # 所有超参数集中放（block_size, batch_size, n_embd...）
├── tokenizer.py   # 词表 + encode/decode（你已开始）
├── data.py        # 读文件、train/val 划分、get_batch
├── model.py       # GPT 模型本体（embedding → attention → blocks → 输出）
├── train.py       # 训练循环（入口脚本）
└── generate.py    # 用训练好的模型生成文本

'''