import numpy as np
from datasets import Dataset
from transformers import BertTokenizer, BertForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, load_metric
from .data_processor import BuddhistPOSDataProcessor

class BuddhistPOSTrainer:
    def __init__(self, model_name="/sikubert", custom_dict_path=None):
        self.processor = BuddhistPOSDataProcessor(model_name, custom_dict_path)
        self.model = None

    def prepare_dataset(self, sentences, train_ratio=0.8):
        """
        准备训练和验证数据集
        """
        tokens, labels = self.processor.prepare_training_data(sentences)

        # 创建数据集
        data = {
            "tokens": tokens,
            "labels": labels
        }
        dataset = Dataset.from_dict(data)

        # 分割训练集和验证集
        dataset = dataset.train_test_split(test_size=1-train_ratio, seed=42)

        # 应用tokenization
        tokenized_dataset = dataset.map(
            self.processor.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        return tokenized_dataset

    def train(self, sentences, output_dir="./buddhist_pos_model", **training_args):
        """
        训练词性标注模型
        """
        # 准备数据
        tokenized_dataset = self.prepare_dataset(sentences)

        # 初始化模型
        self.model = BertForTokenClassification.from_pretrained(
            self.processor.tokenizer.name_or_path,
            num_labels=len(self.processor.label_list),
            id2label=self.processor.id2label,
            label2id=self.processor.label2id
        )

        # 设置训练参数
        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=500,
            report_to=None,
            **training_args
        )

        # 数据收集器
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.processor.tokenizer
        )

        # 评估指标
        metric = load_metric("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [self.processor.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.processor.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        # 训练器
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
            compute_metrics=compute_metrics,
        )

        # 开始训练
        print("开始训练词性标注模型...")
        trainer.train()

        # 保存模型
        trainer.save_model(output_dir)
        self.processor.tokenizer.save_pretrained(output_dir)

        print(f"模型已保存到: {output_dir}")
        return trainer

    def load_model(self, model_path):
        """加载已训练的模型"""
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.processor.tokenizer = BertTokenizer.from_pretrained(model_path)