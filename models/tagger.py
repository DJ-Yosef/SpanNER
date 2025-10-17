import pytorch_lightning as pl
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from models.bert_model_spanner import BertNER
from dataloader.collate_functions import collate_to_max_length

class BertNerTagger(pl.LightningModule):
    def __init__(self, args):
        super(BertNerTagger, self).__init__()
        self.hparams = args

        # 确保 bert_max_length 属性存在
        self.hparams.bert_max_length = getattr(args, "bert_max_length", 128)  # 默认值为 128

        self.bert_model = BertNER.from_pretrained(args.pretrained_bert_model, args=args)
        self.hidden_size = self.bert_model.config.hidden_size
        self.max_span_width = args.max_span_width
        self.tokenLen_emb_dim = getattr(args, "tokenLen_emb_dim", 25)
        self.spanLen_emb_dim = getattr(args, "spanLen_emb_dim", 25)
        self.morph_emb_dim = getattr(args, "morph_emb_dim", 25)

        self._endpoint_span_extractor = EndpointSpanExtractor(
            input_dim=self.hidden_size,
            combination=args.span_combination_mode,  # e.g., "x,y,x*y"
            num_width_embeddings=self.max_span_width,
            span_width_embedding_dim=self.tokenLen_emb_dim,
            bucket_widths=True
        )

        self.spanLen_embedding = nn.Embedding(self.max_span_width + 1, self.spanLen_emb_dim, padding_idx=0)
        self.morph_embedding = nn.Embedding(len(args.morph2idx_list) + 1, self.morph_emb_dim, padding_idx=0)

        # span_embedding初始化为None，forward时自动创建

    def forward(self, input_ids, attention_mask, token_type_ids, span_indices, span_indices_mask, morph_indices):
        # 前向传播逻辑
        sequence_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        span_embeddings = self._endpoint_span_extractor(sequence_output, span_indices, span_indices_mask)
        spanLen_embeddings = self.spanLen_embedding(span_indices_mask.long().sum(dim=-1))
        morph_embeddings = self.morph_embedding(morph_indices)

        # 组合所有嵌入
        combined_embeddings = torch.cat([span_embeddings, spanLen_embeddings, morph_embeddings], dim=-1)

        # 计算输出
        start_outputs = self.start_outputs(combined_embeddings)
        end_outputs = self.end_outputs(combined_embeddings)

        return start_outputs, end_outputs

    def configure_optimizers(self):
        # 优化器配置
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

        num_gpus = len([x for x in str(self.hparams.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.hparams.accumulate_grad_batches * max(1, num_gpus)) + 1) * self.hparams.max_epochs

        if self.hparams.optimizer == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                betas=(0.9, 0.98),
                lr=self.hparams.lr,
                eps=getattr(self.hparams, "adam_epsilon", 1e-8),
            )
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.hparams.lr, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=t_total,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=10000.0,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def get_dataloader(self, split: str):
        dataset = BERTNERDataset(
            data_path=self.hparams.data_path,
            split=split,
            tokenizer=self.tokenizer,
            max_length=self.hparams.bert_max_length,
            possible_only=True,  # 确保只包含有效的 span
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=(split == "train"),
            num_workers=self.hparams.num_workers,
            collate_fn=collate_to_max_length,
        )
        return dataloader

    def prepare_data(self):
        # 准备数据逻辑
        pass

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        span_indices = batch["span_idxs"]
        span_indices_mask = batch["span_lens"]
        morph_indices = batch["morph_idxs"]

        start_outputs, end_outputs = self.forward(input_ids, attention_mask, token_type_ids, span_indices, span_indices_mask, morph_indices)

        # 计算 loss
        start_loss = torch.nn.functional.cross_entropy(
            start_outputs.view(-1, start_outputs.size(-1)),
            labels.view(-1),
            weight=self.loss_weight,
            ignore_index=self.ignore_index,
            reduction="mean"
        )
        end_loss = torch.nn.functional.cross_entropy(
            end_outputs.view(-1, end_outputs.size(-1)),
            labels.view(-1),
            weight=self.loss_weight,
            ignore_index=self.ignore_index,
            reduction="mean"
        )
        loss = start_loss + end_loss

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        span_indices = batch["span_idxs"]
        span_indices_mask = batch["span_lens"]
        morph_indices = batch["morph_idxs"]

        start_outputs, end_outputs = self.forward(input_ids, attention_mask, token_type_ids, span_indices, span_indices_mask, morph_indices)

        # 计算 loss
        start_loss = torch.nn.functional.cross_entropy(
            start_outputs.view(-1, start_outputs.size(-1)),
            labels.view(-1),
            weight=self.loss_weight,
            ignore_index=self.ignore_index,
            reduction="mean"
        )
        end_loss = torch.nn.functional.cross_entropy(
            end_outputs.view(-1, end_outputs.size(-1)),
            labels.view(-1),
            weight=self.loss_weight,
            ignore_index=self.ignore_index,
            reduction="mean"
        )
        loss = start_loss + end_loss

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        span_indices = batch["span_idxs"]
        span_indices_mask = batch["span_lens"]
        morph_indices = batch["morph_idxs"]

        start_outputs, end_outputs = self.forward(input_ids, attention_mask, token_type_ids, span_indices, span_indices_mask, morph_indices)

        # 计算 loss
        start_loss = torch.nn.functional.cross_entropy(
            start_outputs.view(-1, start_outputs.size(-1)),
            labels.view(-1),
            weight=self.loss_weight,
            ignore_index=self.ignore_index,
            reduction="mean"
        )
        end_loss = torch.nn.functional.cross_entropy(
            end_outputs.view(-1, end_outputs.size(-1)),
            labels.view(-1),
            weight=self.loss_weight,
            ignore_index=self.ignore_index,
            reduction="mean"
        )
        loss = start_loss + end_loss

        self.log("test_loss", loss)
        return loss