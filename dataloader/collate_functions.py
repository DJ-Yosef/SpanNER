# encoding: utf-8
import torch
from typing import List

def collate_to_max_length(batch):
    # 当前 batch 中最大 token 序列长度
    max_length = max(x["input_ids"].size(0) for x in batch)

    # 最大 span 数量
    max_num_span = max(len(x["morph_idxs"]) for x in batch)

    # 单个 span 的最大长度
    max_span_len = max(len(morph_idxs[0]) for morph_idxs in (x["morph_idxs"] for x in batch if x["morph_idxs"])) if max_num_span > 0 else 0

    batch_input_ids = []
    batch_attention_mask = []
    batch_token_type_ids = []
    batch_labels = []

    batch_span_idxs = []
    batch_span_weights = []
    batch_span_lens = []
    batch_morph_idxs = []

    for x in batch:
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        token_type_ids = x["token_type_ids"]
        labels = x["labels"]

        span_idxs = x["span_idxs"]
        span_weights = x["span_weights"]
        span_lens = x["span_lens"]
        morph_idxs = x["morph_idxs"]

        pad_len = max_length - input_ids.size(0)

        # input-level padding
        batch_input_ids.append(torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)]))
        batch_attention_mask.append(torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)]))
        batch_token_type_ids.append(torch.cat([token_type_ids, torch.zeros(pad_len, dtype=torch.long)]))
        batch_labels.append(torch.cat([labels, torch.zeros(pad_len, dtype=torch.long)]))

        # span-level padding
        pad_span = max_num_span - len(span_idxs)
        batch_span_idxs.append(torch.tensor(span_idxs + [(0, 0)] * pad_span))
        batch_span_weights.append(torch.tensor(span_weights + [0.0] * pad_span))
        batch_span_lens.append(torch.tensor(span_lens + [0] * pad_span))

        # 处理 morph_idxs 为空的情况
        if morph_idxs:
            batch_morph_idxs.append(torch.tensor(morph_idxs + [[0] * max_span_len for _ in range(pad_span)]))
        else:
            batch_morph_idxs.append(torch.zeros(max_num_span, max_span_len, dtype=torch.long))

    # 调试信息
    print(f"max_length: {max_length}")
    print(f"max_num_span: {max_num_span}")
    print(f"max_span_len: {max_span_len}")
    print(f"batch_morph_idxs: {batch_morph_idxs}")

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "token_type_ids": torch.stack(batch_token_type_ids),
        "labels": torch.stack(batch_labels),
        "span_idxs": torch.stack(batch_span_idxs),
        "span_weights": torch.stack(batch_span_weights),
        "span_lens": torch.stack(batch_span_lens),
        "morph_idxs": torch.stack(batch_morph_idxs),
    }
