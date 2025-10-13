# SpanNER: 基于BERT的中文嵌套命名实体识别模型
##  项目简介
本项目是一个实现 **嵌套实体识别（Nested Named Entity Recognition, NER）** 的中文NLP模型，基于 `BERT` 和 `PyTorch Lightning` 实现，结合 span-based 方法支持复杂结构实体抽取任务。特别适用于处理类书、古文资料、古籍文献等嵌套复杂度较高的语料，基于 [neulab/SpanNER](https://github.com/neulab/SpanNER) 开源实现进行中文适配与结构重构。
##  项目结构
```text
SpanNER/
├── config_spanner.py         # 模型配置类
├── run.py                    # 主入口，控制训练流程
├── trainer.py                # 训练器封装
│
├── models/                   # LightningModule + BertNER 调用
│   ├── tagger.py             # 训练主模块（模型+优化器+数据加载器整合）
│   ├── bert_model_spanner.py # Span-based BERT NER 模型结构
│   └── classifier.py         # 多层分类器（可选）
│
├── dataloader/              # 数据集构造器
│   ├── dataload.py           # 核心数据加载逻辑
│   ├── truncate_dataset.py   # 可选数据裁剪器（限制样本数）
│   └── collate_functions.py  # batch对齐函数
│
├── data/                    # 数据集存放路径
│   ├── train.json            # 训练集
│   ├── dev.json              # 验证集
│   └── test.json             # 测试集
│
├── outputs/                 # 模型训练结果与输出
│   └── epoch_result.txt      # 每轮评估结果保存
│
├── utils/                   # 工具函数（可选）
│   └── random_seed.py        # 随机种子控制
│
├── evaluate.py              # 推理脚本（开发中）
├── metrics.py               # 评估指标计算
└── README.md
```
##  模型工作流程

```
文本输入 → BERT编码 → 生成所有可能的span → 提取特征 → 分类判断 → 输出所有实体（支持嵌套）
```
具体步骤如下：

1. **文本输入与编码**：原始文本通过 tokenizer 编码成 token 序列，输入 BERT 得到每个 token 的上下文向量表示。
2. **枚举 span 片段**：对每一句话中的所有起止 token 组合生成可能的 span（子串）。
3. **特征拼接与增强**：对每个 span，提取起点与终点的向量，并拼接长度嵌入、形态特征等（如是否是数字、大写等）。
4. **分类判断**：用一个全连接网络（如线性层）判断每个 span 是不是一个实体、属于哪一类。
5. **支持嵌套结构**：由于每个 span 独立判断，模型可识别重叠的多个实体，如“[唐朝诗人李白]” 和 “[李白]”。
 示例：句子“唐朝诗人李白生于四川”
模型会识别：
- “李白” → 人物
- “唐朝诗人” → 职业称谓
- “四川” → 地点
- “唐朝诗人李白” → 嵌套结构

因此特别适合处理中文类书、古籍和医学文本中的复杂嵌套结构。
（翻译）
##  依赖安装
```bash
pip install -r requirements.txt
```
或使用 Conda 虚拟环境：
```bash
conda create -n spanner python=3.9（注意是3.9，否则会报错）
conda activate spanner
pip install torch pytorch-lightning transformers
```
##  快速启动训练
确保已准备好以下内容：
- `data/train.json`, `data/dev.json`, `data/test.json`：三个标注好的数据集，格式为自定义 span 格式（详见 dataload.py）。
- 可选：你也可以使用 `truncate_dataset.py` 快速构造子集用于调试。
运行命令：
```bash
python run.py
```
## 预训练模型说明
默认使用 `bert-base-chinese`。如需使用更优的古文预训练模型（如复旦大学发布的 `guwenBERT`），请在[哈工大语言技术平台](https://huggingface.co/IDEA-CCNL/Erlangshen-GuwenBERT)登录后下载。
替换方式：
```python
bert_config_dir = "路径/guwenBERT"
```
模型文件需要包含：`config.json`, `pytorch_model.bin`, `vocab.txt`。

##  模型亮点
- 支持中文嵌套实体识别（Nested NER）
- 支持 morph 特征（例如：全大写、标题、数字等）
- 自定义 span 结构与组合策略（如 `x, y, x*y`）
- 全部模块可控，基于 `Lightning` 构建，便于调试与复用

##  常见错误说明
| 错误信息关键词           | 解决方法 |
|-------------------------|----------|
| `bert_config_dir`       | 检查 run.py 中路径是否正确 |
| `AdamW deprecated`      | 可忽略，也可切换至 `torch.optim.AdamW` |
| `spanner.dev not found` | 确保 `data/` 下包含正确的数据文件 |


FAQ：遇到数据格式/规格问题怎么办？
1. 最常见问题
Padding/Label shape mismatch（padding或标签形状对不齐）
实体 start/end 索引越界
字段缺失或 json 解析报错
2. 需要修改的主要文件
作用	文件路径	主要修改内容
数据预处理	dataloader/dataload.py	调整读取/解析数据格式，保证每个样本都是 dict，且有 text 和 entities 字段
collate 处理函数	dataloader/collate_functions.py	控制 batch 拼接、padding、span 等 shape 对齐（遇到 shape 报错优先改这里）
数据样例检查	你的训练/验证集 json 数据本身	检查每条 json 格式、实体 span start/end 不要超出 text 长度
3. 怎么排查/定位？
看到 “mat1 and mat2 shapes cannot be multiplied”、“batch_size 不一致”等错误，第一步看 collate_functions.py 里返回的所有 shape。
如果模型能加载但报 entity 相关的 IndexError，看 dataload.py 的数据样本生成那步。
不知道哪里出错，就直接加 print() 打印每个 batch 的数据 shape。
4. 遇到具体错误怎么办？
先定位报错行，结合上表找到负责对应操作的 py 文件，按照注释和上面标准数据格式对照修正。
只改数据或 collate，不动模型结构！
总结
遇到数据/格式相关的报错（如 shape 不一致、实体 span 对不齐），请重点检查并修改：
dataloader/dataload.py（数据解析/预处理）
dataloader/collate_functions.py（collate 和 padding 对齐）
只要你的数据文件、collate 返回的 tensor 维度都是一致且合理的，这类 shape 报错基本都能解决。

