# encoding: utf-8
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertPreTrainedModel, RobertaModel
from .classifier import MultiNonLinearClassifier, SingleLinearClassifier
from allennlp.modules.span_extractors import EndpointSpanExtractor

class BertNER(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertNER, self).__init__(config)
        self.args = args

        if 'roberta' in self.args.bert_config_dir.lower():
            self.bert = RobertaModel(config)
            print('→ 使用 RoBERTa 预训练模型')
        else:
            self.bert = BertModel(config)
            print('→ 使用 BERT 预训练模型')

        self.hidden_size = config.hidden_size
        self.n_class = args.n_class
        self.max_span_width = args.max_spanLen

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
        self.span_embedding = None

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)

    def forward(self, span_weights, span_lens, span_idxs,
                input_ids, attention_mask=None, token_type_ids=None,
                morph_idxs=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (bs, seq_len, hidden_size)

        span_feats = self._endpoint_span_extractor(sequence_output, span_idxs.long())
        features = [span_feats]

        if self.args.use_spanLen:
            spanlen_emb = F.relu(self.spanLen_embedding(span_lens))
            features.append(spanlen_emb)

        if self.args.use_morph and morph_idxs is not None:
            morph_emb = self.morph_embedding(morph_idxs)
            morph_sum = torch.sum(morph_emb, dim=2)  # sum pooling
            features.append(morph_sum)

        span_feature = torch.cat(features, dim=-1)
        print(f"→ Forward 拼接后 span_feature.shape[-1] = {span_feature.shape[-1]}")

        # 动态创建 MultiNonLinearClassifier
        if self.span_embedding is None:
            print(f"→ 动态创建span分类器，输入维度 = {span_feature.shape[-1]}")
            self.span_embedding = MultiNonLinearClassifier(
                span_feature.shape[-1], self.n_class, self.args.model_dropout
            )
            # 放到同设备（否则GPU/CPU会报错）
            self.span_embedding = self.span_embedding.to(span_feature.device)

        span_logits = self.span_embedding(span_feature)  # 每个 span 的类别分布
        return span_logits
