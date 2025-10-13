# encoding: utf-8

import torch.nn as nn
import torch.nn.functional as F

class SingleLinearClassifier(nn.Module):
    """
    简单的线性分类器，用于最基础的分类任务。
    输入维度：input_dim
    输出维度：num_label
    """
    def __init__(self, input_dim, num_label):
        super(SingleLinearClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_label)

    def forward(self, input_features):
        return self.classifier(input_features)

class MultiNonLinearClassifier(nn.Module):
    """
    多层非线性分类器：
    input_dim -> GELU -> Dropout -> input_dim -> num_label
    适合复杂任务（如span分类/拼接特征）。
    """
    def __init__(self, input_dim, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.classifier1 = nn.Linear(input_dim, input_dim)
        self.classifier2 = nn.Linear(input_dim, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        x = self.classifier1(input_features)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.classifier2(x)
