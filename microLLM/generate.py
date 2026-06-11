#用自回归的思想来实现生成
#生成当前序列，预测下一个字，新序列预测下一个字
import torch.nn as nn
class Generate(nn.Module):
    