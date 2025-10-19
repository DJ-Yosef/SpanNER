# encoding: utf-8

import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer  # 使用 transformers 版本的 tokenizer

class BERTNERDataset(Dataset):
    def __init__(
        self,
        args,
        json_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 128,
        possible_only: bool = False,
        pad_to_maxlen: bool = False
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_maxlen = pad_to_maxlen
        self.possible_only = possible_only
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.labels = ['unknown', 'breed', 'usage', 'concept', 'skill', 'tool']

        if self.possible_only:
            self.all_data = [x for x in self.all_data if x.get("start_position")]

        self.max_span_len = self.args.max_spanLen
        minus = (self.max_span_len + 1) * self.max_span_len // 2
        self.max_num_span = self.max_length * self.max_span_len - minus
        self.dataname = self.args.dataname
        self.spancase2idx_dic = {}

        self.empty_ner_list = 0

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        """获取数据集中的一个样本时，自动检查格式错误，并构造输入、标签、遮罩等数据，和原文一起返回。"""
        data = self.all_data[idx]
        context = data["sentences"].strip()
        context = context.replace("\u200b", "").replace("\ufeff", "").replace("　", " ")

        # 支持 ["实体", "标签", [start, end]] 格式
        ner_list = data.get("ner", [])
        pos_span_idxs = []

        valid_entities = 0

        span_labels = []
        try:  # 处理非法的 span
            for text, label, span in ner_list:
                start, end = span
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError(f"无效 span: ({start}, {end}) in label={label}")
                if context[start:end + 1] != text:
                    raise ValueError(f"实体 {text} 与上下文不匹配: {context[start:end + 1]}")
                pos_span_idxs.append((start, end))
                valid_entities += 1
                if label not in self.labels:
                    span_labels.append(0)
                else:
                    span_labels.append(self.labels.index(label) + 1)
        except Exception as e:
            print(f"出错样本 index: {idx}")
            print(f"内容片段: {context}")
            print(f"ner_list: {ner_list}")
            raise e

        if valid_entities == 0:
            self.empty_ner_list += 1
            print(f"出错样本 index: {idx}")
            print(f"内容片段: {context}")
            print(f"ner_list: {ner_list}")
            print(f'发现{self.empty_ner_list}个')
            # raise ValueError("没有有效的实体")

        all_span_idxs = pos_span_idxs
        all_span_weights = [1.0] * len(all_span_idxs)  # 权重默认全部为 1.0
        all_span_lens = [int(e) - int(s) + 1 for s, e in all_span_idxs]  # 实体长度
        if not all_span_idxs:
            # 如果没有实体，创建一个空的但具有正确维度的 morph_idxs 和 span_labels
            morph_idxs = []
            # span_labels = []
        else:
            morph_idxs = [[0] * self.max_span_len for _ in all_span_idxs]  # 生成默认全 0 的向量
        # 将列表转换为张量
        span_labels = torch.tensor(span_labels, dtype=torch.long) if span_labels else torch.tensor([], dtype=torch.long)

        # 使用 transformers 的 encode_plus 将句子编码为 BERT 所需的格式
        # 生成 输入, 遮罩, 标签
        encoded = self.tokenizer.encode_plus(
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',  # 保证 input_ids 是固定长度
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors=None
        )

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(encoded["token_type_ids"], dtype=torch.long)
        labels = torch.zeros(self.max_length, dtype=torch.long)  # 默认全0作为伪标签

        # # 填充 labels
        # for (start, end), label in zip(all_span_idxs, all_span_lens):
        #     start_idx = encoded["input_ids"].index(self.tokenizer.cls_token_id) + 1 + start
        #     end_idx = start_idx + label - 1
        #     if start_idx < self.max_length and end_idx < self.max_length:
        #         labels[start_idx:end_idx + 1] = self.args.label2idx.get(label, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            # "labels": labels,
            "labels": span_labels,

            "span_idxs": all_span_idxs,
            "span_weights": all_span_weights,
            "span_lens": all_span_lens,
            "morph_idxs": morph_idxs,
        }

    def pad(self, lst, value=0, max_length=None):
        """用指定值 value 填充列表到指定长度"""
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst

    def case_feature_tokenLevel(self, morph2idx, span_idxs, words):
        """特征生成函数：
        生成每个命名实体跨度的形态特征向量，如是否为大写、小写、标题等。
        需要传入一个形态特征到索引的映射字典 morph2idx、命名实体的跨度索引 span_idxs 和原始句子分词后的 words。"""
        pad_len = self.max_span_len
        morph_vec = []
        for s, e in span_idxs:
            vec = [0] * pad_len
            for i, token in enumerate(words[s:e + 1]):
                if token.isupper():
                    vec[i] = morph2idx.get("isupper", 0)
                elif token.islower():
                    vec[i] = morph2idx.get("islower", 0)
                elif token.istitle():
                    vec[i] = morph2idx.get("istitle", 0)
                elif token.isdigit():
                    vec[i] = morph2idx.get("isdigit", 0)
                else:
                    vec[i] = morph2idx.get("other", 0)
            morph_vec.append(vec)
        return morph_vec

    def convert2tokenIdx(self, words, tokens, type_ids, offsets, span_idxs, span_idxLab):
        max_len = self.max_length
        sidxs = [s + sum(len(w) for w in words[:s]) for s, _ in span_idxs]
        eidxs = [e + sum(len(w) for w in words[:e + 1]) for _, e in span_idxs]

        span_new_label = {}
        for (s, e), os_str in zip(zip(sidxs, eidxs), span_idxs):
            k = f"{os_str[0]};{os_str[1]}"
            span_new_label[f"{s};{e}"] = span_idxLab.get(k, 'O')

        offset2sidx = {s: i for i, (s, e) in enumerate(offsets) if s != 0 or e != 0}
        offset2eidx = {e: i for i, (s, e) in enumerate(offsets) if s != 0 or e != 0}

        span_token_idxs = []
        valid_span_words = []
        n = 0
        for s, e in zip(sidxs, eidxs):
            if s in offset2sidx and e in offset2eidx and offset2eidx[e] < max_len:
                span_token_idxs.append((offset2sidx[s], offset2eidx[e]))
                valid_span_words.append(words[span_idxs[n][0]:span_idxs[n][1] + 1])
            n += 1
        return span_token_idxs, valid_span_words, span_new_label
        print(f"加载验证样本总数：{len(self.samples)}")