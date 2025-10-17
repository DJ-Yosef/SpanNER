import os
import torch
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torch.optim import SGD
from transformers import BertTokenizer #, AdamW #AdamW 在4.5.0版本中被移除
from torch.optim import AdamW

from dataloader.dataload import BERTNERDataset
from dataloader.truncate_dataset import TruncateDataset
from dataloader.collate_functions import collate_to_max_length
from models.bert_model_spanner import BertNER
from config_spanner import BertNerConfig

class BertNerTagger(LightningModule):
    def __init__(self, args=None, **kwargs):
        super().__init__()

        if args is not None:
            self.save_hyperparameters(vars(args))
        else:
            self.save_hyperparameters(kwargs)

        self.bert_dir = self.hparams.bert_config_dir
        self.data_dir = self.hparams.data_dir

        bert_config = BertNerConfig.from_pretrained(
            self.bert_dir,
            hidden_dropout_prob=self.hparams.model_dropout,
            attention_probs_dropout_prob=self.hparams.model_dropout,
            model_dropout=self.hparams.model_dropout
        )

        self.model = BertNER.from_pretrained(
            self.bert_dir,
            config=bert_config,
            args=self.hparams
        )

        self.optimizer = self.hparams.optimizer
        self.n_class = self.hparams.n_class
        self.max_spanLen = self.hparams.max_spanLen
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.classifier = torch.nn.Softmax(dim=-1)

        self.fwrite_epoch_res = open(self.hparams.fp_epoch_result, 'w', encoding='utf-8')
        self.fwrite_epoch_res.write("f1, recall, precision, correct_pred, total_pred, total_golden\n")

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                betas=(0.9, 0.98),
                lr=self.hparams.lr,
                eps=getattr(self.hparams, "adam_epsilon", 1e-8),
            )
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.hparams.lr, momentum=0.9)

        num_gpus = len([x for x in str(self.hparams.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.hparams.accumulate_grad_batches * max(1, num_gpus)) + 1) * self.hparams.max_epochs

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            pct_start=self.hparams.warmup_ratio,
            final_div_factor=self.hparams.final_div_factor,
            total_steps=t_total,
            anneal_strategy='linear'
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, span_weights, span_lens, span_idxs, input_ids, attention_mask, token_type_ids, morph_idxs=None):
        return self.model(span_weights, span_lens, span_idxs, input_ids, attention_mask, token_type_ids, morph_idxs)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        span_idxs = batch["span_idxs"]
        span_lens = batch["span_lens"]
        span_weights = batch["span_weights"]
        morph_idxs = batch.get("morph_idxs", None)

        logits = self.forward(span_weights, span_lens, span_idxs, input_ids, attention_mask, token_type_ids, morph_idxs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.n_class),
            labels.view(-1),
            ignore_index=-100
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        span_idxs = batch["span_idxs"]
        span_lens = batch["span_lens"]
        span_weights = batch["span_weights"]
        morph_idxs = batch.get("morph_idxs", None)

        logits = self.forward(span_weights, span_lens, span_idxs, input_ids, attention_mask, token_type_ids, morph_idxs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.n_class),
            labels.view(-1),
            ignore_index=-100
        )
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        json_path = os.path.join(self.data_dir, f"spanner.{prefix}")
        tokenizer = BertTokenizer.from_pretrained(self.bert_dir)
        dataset = BERTNERDataset(
            self.hparams,
            json_path=json_path,
            tokenizer=tokenizer,
            max_length=self.hparams.bert_max_length,
            pad_to_maxlen=False
        )
        if limit is not None:
            dataset = TruncateDataset(dataset, limit)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True if prefix == "train" else False,
            drop_last=False,
            collate_fn=collate_to_max_length
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def test_dataloader(self):
        return self.get_dataloader("test")