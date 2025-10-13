# encoding: utf-8

from transformers import BertConfig


class BertNerConfig(BertConfig):
    """
    在 BertConfig 基础上扩展的自定义配置类。
    可支持如 model_dropout 等额外参数。
    """
    def __init__(self, **kwargs):
        # 先调用父类初始化，加载基础参数
        super(BertNerConfig, self).__init__(**kwargs)

        # 自定义配置：可按需添加更多超参数
        self.model_dropout = kwargs.pop("model_dropout", 0.1)
