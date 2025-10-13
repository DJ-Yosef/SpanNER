# encoding: utf-8
import os
import sys
from argparse import Namespace
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_spanner import BertNerConfig
from transformers import BertTokenizer
from trainer import train
# 将参数改为 dict 或 Namespace，确保可以被 LightningModule 接收
args = Namespace(
    # ===== 数据参数 =====
    bert_max_length=128,
    train_file="data/train.json",
    dev_file="data/dev.json",
    test_file="data/test.json",
    data_dir="data",
    dataname="spanner",
    # ===== 模型结构参数 =====
    # bert_config_dir="pretrained_model\guwenbert-base",
    bert_config_dir="pretrained_model\bert-base-chinese",
    n_class=4,
    max_spanLen=10,
    tokenLen_emb_dim=25,
    spanLen_emb_dim=25,
    morph_emb_dim=25,
    use_spanLen=True,
    use_morph=True,
    span_combination_mode="x,y,x*y",
    model_dropout=0.1,
    # ===== 输出路径 =====
    fp_epoch_result="outputs/epoch_result.txt",
    # ===== 训练控制参数 =====
    batch_size=8,
    epochs=5,
    max_epochs=10,
    lr=3e-5,
    gpus=0,  # 如果用GPU：设置为 accelerator="gpu", devices=1
    accumulate_grad_batches=1,
    warmup_steps=200,
    final_div_factor=1e4,
    # ===== 优化器与调度器 =====
    optimizer="adamw",  # 注意大小写，与内部判断保持一致
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    scheduler="linear",
    # ===== 额外嵌入特征（词形）=====
    morph2idx_list=["None", "AllUpper", "AllLower", "Title", "Digit"],
    adam_epsilon=1e-8  # 避免 optimizer 报错
)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(args.fp_epoch_result), exist_ok=True)

    print("→ 使用 BERT 预训练模型")
    config = BertNerConfig.from_pretrained(args.bert_config_dir)
    tokenizer = BertTokenizer.from_pretrained(args.bert_config_dir)

    train(args, config, tokenizer)
