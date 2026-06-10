# 将数据化为token

class Tokenizer:
    # 1) 建词表：找出文本里出现过的所有不同字符，排序后固定下来
    #set(text) 去重，sorted 让顺序稳定（每次跑结果一致）
    def __init__(self, text: str):
        #准备去重
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        #建立映射关系
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        print("\nvocab_size（不同字符数）:", self.vocab_size)
        print("所有字符:", "".join(self.chars))
        

    # 2) encode：字符串 -> 数字列表 ；decode：数字列表 -> 字符串
    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, nums: list[int]):
        return "".join(self.itos[i] for i in nums)
