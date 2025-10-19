import json
import os
# 读取原始数据
with open('dataprocess/converted_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

text = data['text']
entities = data['entities']

# 读取示例格式的数据以便参考
with open('data/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 示例格式中的每个句子的最大长度（以token计）
max_tokens_per_sentence = 64

# 分割文本为句子
import re
sentences = re.split(r'(?<=[。\n])', text)

# 将句子和实体映射为新的格式
new_data = []
current_index = 0
labels = set([entity['label'] for entity in entities])
print(f"识别到的实体标签有: {labels}")

for sentence in sentences:
    sentence_entities = []
    for entity in entities:
        # 如果实体在当前句子中
        if current_index <= entity['start'] < current_index + len(sentence):
            entity_start = entity['start'] - current_index
            entity_end = entity['end'] - current_index
            sentence_entities.append([entity['text'], entity['label'], [entity_start, entity_end-1]])
    # 如果句子不为空，添加到新的数据列表中
    if sentence.strip() and sentence_entities != []:
        new_data.append({
            "sentences": sentence.strip(),
            "ner": sentence_entities
        })
    current_index += len(sentence)

# 确保每个句子的长度不超过最大token数
max_tokens_per_sentence = 128
final_data = []
for item in new_data:
    if len(item["sentences"].split()) <= max_tokens_per_sentence:
        final_data.append(item)
    else:
        # 如果句子过长，需要进一步分割
        words = item["sentences"].split()
        start_index = 0
        current_ner = item["ner"]
        while start_index < len(words):
            end_index = start_index + max_tokens_per_sentence
            sub_sentence = ' '.join(words[start_index:end_index])
            sub_sentence_entities = [
                [e[0], e[1], [e[2][0] - start_index, e[2][1] - start_index]]
                for e in current_ner if start_index <= e[2][0] < end_index
            ]
            if sub_sentence.strip() and sub_sentence_entities != []:
                final_data.append({
                    "sentences": sub_sentence.strip(),
                    "ner": sub_sentence_entities
                })
            start_index = end_index

print(len(final_data))
final_data = [i for i in final_data if i['ner']!= []]
print(len(final_data))

# 写入新的JSON格式文件
with open('dataprocess/converted_train.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print("转换完成，结果保存在 dataprocess/converted_train.json 文件中")

import numpy as np
from random_seed import set_random_seed
set_random_seed(42)

sample_idx = list(range(len(final_data)))
np.random.shuffle(sample_idx)
## 按照7:2:1划分训练集、验证集和测试集
train_split = int(0.7 * len(final_data))
dev_split = int(0.9 * len(final_data))
train_idx = sample_idx[:train_split]
dev_idx = sample_idx[train_split:dev_split]
test_idx = sample_idx[dev_split:]

os.makedirs('data/test1', exist_ok=True)
with open('data/test1/spanner.train', 'w', encoding='utf-8') as f:
    json.dump([final_data[i] for i in train_idx], f, ensure_ascii=False, indent=2)
with open('data/test1/spanner.dev', 'w', encoding='utf-8') as f:
    json.dump([final_data[i] for i in dev_idx], f, ensure_ascii=False, indent=2)
with open('data/test1/spanner.test', 'w', encoding='utf-8') as f:
    json.dump([final_data[i] for i in test_idx], f, ensure_ascii=False, indent=2)

print("数据集划分完成，训练集、验证集和测试集保存在 data/test1 目录中")