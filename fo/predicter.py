import torch
from transformers import BertTokenizer, BertForTokenClassification
from tqdm import tqdm

class BuddhistPOSPredictor:
    def __init__(self, model_path):
        """
        初始化预测器
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.model.eval()

        # 获取标签映射
        self.id2label = self.model.config.id2label

    def predict(self, text):
        """
        对文本进行词性标注
        """
        # Tokenize输入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)

        # 处理预测结果
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[pred.item()] for pred in predictions[0]]

        # 对齐原始文本
        results = self._align_predictions(text, tokens, predicted_labels, inputs["offset_mapping"][0])

        return results

    def _align_predictions(self, text, tokens, labels, offset_mapping):
        """
        将预测结果与原始文本对齐
        """
        results = []
        current_word = ""
        current_label = ""

        for token, label, offset in zip(tokens, labels, offset_mapping):
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token]:
                continue

            start, end = offset
            if start == end == 0:
                continue

            char = text[start:end]

            # 处理subword tokens
            if token.startswith("##"):
                current_word += char
            else:
                if current_word:
                    results.append((current_word, current_label))
                current_word = char
                current_label = label

        if current_word:
            results.append((current_word, current_label))

        return results

    def batch_predict(self, texts, batch_size=8):
        """
        批量预测
        """
        all_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="批量预测"):
            batch_texts = texts[i:i+batch_size]
            batch_results = [self.predict(text) for text in batch_texts]
            all_results.extend(batch_results)

        return all_results