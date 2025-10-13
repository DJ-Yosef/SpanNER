# encoding: utf-8

from torch.utils.data import Dataset


class TruncateDataset(Dataset):
    """
    用于截断原始数据集，仅使用前 max_num 个样本。

    典型应用场景：
    - 模型调试、测试数据加载是否正常
    - 快速过小数据集进行 sanity check
    - 减少训练时间用于快速迭代
    """

    def __init__(self, dataset: Dataset, max_num: int = 100):
        """
        参数:
            dataset: 原始 Dataset 对象
            max_num: 最多使用的样本数量，默认最多保留 100 条
        """
        self.dataset = dataset
        self.max_num = min(max_num, len(self.dataset))  # 避免超过实际长度

    def __len__(self):
        return self.max_num  # 返回截断后的样本数

    def __getitem__(self, item):
        return self.dataset[item]  # 委托给原数据集取样本

    def __getattr__(self, item):
        """
        保留原始 dataset 的方法和属性访问（例如 tokenizer、label2idx 等）
        """
        return getattr(self.dataset, item)
