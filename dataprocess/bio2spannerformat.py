# encoding: utf-8
import os
import json
import codecs
from collections import Counter


def get_chunk_type(tag: str):
    tag_class = tag.split('-')[0]
    tag_type = '-'.join(tag.split('-')[1:])
    return tag_class, tag_type


def get_chunks(seq):
    default = 'O'
    chunks = []
    chunk_type, chunk_start = None, None

    for i, tag in enumerate(seq):
        if tag == default:
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = None, None
        else:
            tag_class, tag_type = get_chunk_type(tag)
            if chunk_type is None:
                chunk_type, chunk_start = tag_type, i
            elif tag_type != chunk_type or tag_class == "B":
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = tag_type, i

    if chunk_type is not None:
        chunks.append((chunk_type, chunk_start, len(seq)))
    return chunks


def read_conll_data(filepath, column_no=-1, delimiter=' ', corpus_type='conll03'):
    word_sequences = []
    tag_sequences = []
    sentence_words = []
    sentence_tags = []

    with codecs.open(filepath, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if "❤ ️" in line:
                line = line.replace("❤ ️", "[emoji]")
            if line == "":
                if sentence_words:
                    word_sequences.append(sentence_words)
                    tag_sequences.append(sentence_tags)
                    sentence_words, sentence_tags = [], []
                continue

            splits = line.split(delimiter)
            if len(splits) <= column_no:
                continue
            word = splits[0].strip()
            tag = splits[column_no].strip()

            if corpus_type == 'ptb2':
                tag = 'B-' + tag
            if word == '❤':
                word = '[emoji]'

            sentence_words.append(word)
            sentence_tags.append(tag)

        if sentence_words:
            word_sequences.append(sentence_words)
            tag_sequences.append(sentence_tags)

    return word_sequences, tag_sequences


def convert_to_spanner_format(words_list, tags_list):
    all_labels = []
    for tags in tags_list:
        for tag in tags:
            if tag != 'O':
                parts = tag.split('-')
                label = '-'.join(parts[1:]) if len(parts) > 2 else parts[-1]
                all_labels.append(label)

    label_counter = Counter(all_labels)
    label2idx = {"O": 0}
    for i, (label, _) in enumerate(label_counter.items(), start=1):
        label2idx[label] = i

    data = []
    for tokens, tags in zip(words_list, tags_list):
        chunks = get_chunks(tags)
        context = ' '.join(tokens)
        span_labels = {
            f"{start};{end - 1}": label
            for label, start, end in chunks
        }
        sample = {
            "context": context,
            "span_posLabel": span_labels
        }
        data.append(sample)
    return data, label2idx


def convert_all_files(dataset_name, input_dir, output_dir, suffix_list=["train", "dev", "test"]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset_name == 'ontonote5':
        column_no, delimiter = 3, ' '
    elif 'wnut' in dataset_name:
        column_no, delimiter = 1, '\t'
    else:
        column_no, delimiter = -1, ' '

    for suffix in suffix_list:
        input_file = os.path.join(input_dir, f"{suffix}.txt")
        words, tags = read_conll_data(input_file, column_no=column_no, delimiter=delimiter, corpus_type=dataset_name)
        data, label2idx = convert_to_spanner_format(words, tags)
        output_file = os.path.join(output_dir, f"spanner.{suffix}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {output_file}")

    print(f"[✓] label2idx: {label2idx}")
    return label2idx


if __name__ == '__main__':
    dataname = 'conll03'
    input_path = '../data/conll03_bio'
    output_path = '../data/conll03_spanner'
    convert_all_files(dataname, input_path, output_path)
