# encoding: utf-8
import argparse
import os
from typing import Dict
import pytorch_lightning as pl
import torch
from models.tagger import BertNerTagger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import BertTokenizer
from torch.optim import AdamW
from torch.optim import SGD
from torch import Tensor
from torch.utils.data import DataLoader
import pickle
import random
import logging

from dataloader.dataload import BERTNERDataset
from dataloader.truncate_dataset import TruncateDataset
from dataloader.collate_functions import collate_to_max_length

from models.bert_model_spanner import BertNER
from config_spanner import BertNerConfig
from random_seed import set_random_seed
#from evaluate import span_f1, span_f1_prune, get_predict, get_predict_prune 到时候可以加上

logger = logging.getLogger(__name__)
set_random_seed(0)
def train(args, config, tokenizer):
    model = BertNerTagger(args)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        # gpus=args.gpus,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=False
    )
    trainer.fit(model)
