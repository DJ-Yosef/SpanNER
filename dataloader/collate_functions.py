# encoding: utf-8
import torch
from typing import List

def collate_to_max_length(batch):
    # 当前 batch 中最大 token 序列长度
    max_length = max(x["input_ids"].size(0) for x in batch)

    # 最大 span 数量
    max_num_span = max(len(x["span_idxs"]) for x in batch)

    # 使用固定的 max_span_len
    max_span_len = 10  # 与数据集中的 self.max_span_len 保持一致

    batch_input_ids = []
    batch_attention_mask = []
    batch_token_type_ids = []
    batch_labels = []  # 现在存储span级别的标签

    batch_span_idxs = []
    batch_span_weights = []
    batch_span_lens = []
    batch_morph_idxs = []

    for x in batch:
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        token_type_ids = x["token_type_ids"]
        labels = x["labels"]  # span级别的标签

        span_idxs = x["span_idxs"]
        span_weights = x["span_weights"]
        span_lens = x["span_lens"]
        morph_idxs = x["morph_idxs"]

        pad_len = max_length - input_ids.size(0)

        # input-level padding
        batch_input_ids.append(torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)]))
        batch_attention_mask.append(torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)]))
        batch_token_type_ids.append(torch.cat([token_type_ids, torch.zeros(pad_len, dtype=torch.long)]))

        # span-level padding
        pad_span = max_num_span - len(span_idxs)
        batch_span_idxs.append(torch.tensor(span_idxs + [(0, 0)] * pad_span))
        batch_span_weights.append(torch.tensor(span_weights + [0.0] * pad_span))
        batch_span_lens.append(torch.tensor(span_lens + [0] * pad_span))

        # 修复 morph_idxs 处理
        if not morph_idxs:
            morph_idxs = []
        batch_morph_idxs.append(torch.tensor(morph_idxs + [[0] * max_span_len for _ in range(pad_span)]))

        # 对span标签进行填充，使用-100作为忽略索引
        if len(labels.shape) == 0:  # 如果是标量
            labels_padded = torch.full((max_num_span,), -100, dtype=torch.long)
            if max_num_span > 0:
                labels_padded[0] = labels
        else:  # 如果是向量
            labels_padded = torch.cat([labels, torch.full((pad_span,), -100, dtype=torch.long)])
        batch_labels.append(labels_padded)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "token_type_ids": torch.stack(batch_token_type_ids),
        "labels": torch.stack(batch_labels),  # 现在所有标签都是相同长度
        "span_idxs": torch.stack(batch_span_idxs),
        "span_weights": torch.stack(batch_span_weights),
        "span_lens": torch.stack(batch_span_lens),
        "morph_idxs": torch.stack(batch_morph_idxs),
    }