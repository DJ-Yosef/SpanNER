import torch
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import jieba
import jieba.posseg as pseg
import re

class BuddhistPOSDataProcessor:
    def __init__(self, model_name="model/sikubert", custom_dict_path=None):
        """
        初始化佛教文本词性标注数据处理器

        参数:
        model_name: SikuBERT模型名称
        custom_dict_path: 自定义佛教词典路径
        """
        # 加载SikuBERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # 加载自定义词典
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
            print(f"已加载自定义词典: {custom_dict_path}")

        # 定义词性标签映射
        self.label_list = [
            'O',  # 其他
            'B-N', 'I-N',  # 名词
            'B-V', 'I-V',  # 动词
            'B-ADJ', 'I-ADJ',  # 形容词
            'B-ADV', 'I-ADV',  # 副词
            'B-P', 'I-P',  # 介词
            'B-C', 'I-C',  # 连词
            'B-PRON', 'I-PRON',  # 代词
            'B-NUM', 'I-NUM',  # 数词
            'B-BUD', 'I-BUD',  # 佛教专有名词
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

        # 扩展佛教专有词性标签
        self._extend_buddhist_labels()

    def _extend_buddhist_labels(self):
        """扩展佛教专有词性标签"""
        # 佛教特有的词性类别
        buddhist_specific = [
            'B-BUD-TERM', 'I-BUD-TERM',  # 佛教术语
            'B-BUD-SUTRA', 'I-BUD-SUTRA',  # 佛经名称
            'B-BUD-PERSON', 'I-BUD-PERSON',  # 佛教人物
            'B-BUD-PLACE', 'I-BUD-PLACE',  # 佛教地点
        ]

        for label in buddhist_specific:
            if label not in self.label2id:
                idx = len(self.label_list)
                self.label_list.append(label)
                self.label2id[label] = idx
                self.id2label[idx] = label

    def convert_jieba_pos_to_ner(self, jieba_pos):
        """将jieba词性标签转换为NER标签"""
        pos_mapping = {
            'n': 'BUD-TERM',  # 名词->佛教术语
            'v': 'V',         # 动词
            'a': 'ADJ',       # 形容词
            'd': 'ADV',       # 副词
            'p': 'P',         # 介词
            'c': 'C',         # 连词
            'r': 'PRON',      # 代词
            'm': 'NUM',       # 数词
            # 佛教专有名词识别
            'nh': 'BUD-PERSON',  # 人名->佛教人物
            'ni': 'BUD-PLACE',   # 机构名->佛教地点
            'ws': 'BUD-SUTRA',   # 外文译名->佛经名称
        }
        return pos_mapping.get(jieba_pos, 'O')

    def prepare_training_data(self, sentences, use_jieba_first=True):
        """
        准备训练数据

        参数:
        sentences: 句子列表
        use_jieba_first: 是否先用jieba进行初步标注
        """
        tokenized_sentences = []
        labels = []

        for sentence in tqdm(sentences, desc="准备训练数据"):
            if use_jieba_first:
                # 使用jieba进行初步分词和词性标注
                words_with_pos = pseg.cut(sentence)
                words, pos_tags = zip(*[(word, pos) for word, pos in words_with_pos])

                # 转换为NER格式
                sentence_tokens = []
                sentence_labels = []

                for word, pos in zip(words, pos_tags):
                    word_tokens = self.tokenizer.tokenize(word)
                    ner_label = self.convert_jieba_pos_to_ner(pos)

                    if len(word_tokens) == 0:
                        continue

                    # 第一个token用B-，后面的用I-
                    sentence_tokens.extend(word_tokens)
                    sentence_labels.append(f"B-{ner_label}")
                    for _ in range(len(word_tokens) - 1):
                        sentence_labels.append(f"I-{ner_label}")

            else:
                # 直接按字符切分（用于人工标注的数据）
                sentence_tokens = [char for char in sentence if char.strip()]
                sentence_labels = ['O'] * len(sentence_tokens)

            if sentence_tokens:
                tokenized_sentences.append(sentence_tokens)
                labels.append(sentence_labels)

        return tokenized_sentences, labels

    def tokenize_and_align_labels(self, examples):
        """
        Tokenize并对齐标签
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            padding=True,
            is_split_into_words=True,
            max_length=512,
            return_tensors="pt"
        )

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[label[word_idx]])
                else:
                    label_ids.append(self.label2id[label[word_idx]])
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs