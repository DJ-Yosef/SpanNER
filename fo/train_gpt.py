import os
import torch
import logging
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torchcrf import CRF
from torch import nn
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "model/sikubert"  # 可改为 SikuBERT
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------- Dataset 读取 ------------------------
class POSDataset(Dataset):
    def __init__(self, path, tokenizer, label2id):
        self.texts, self.labels = self.read_data(path)
        self.tokenizer = tokenizer
        self.label2id = label2id

    def read_data(self, path):
        texts, labels, text, label = [], [], [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if text:
                        texts.append(text)
                        labels.append(label)
                        text, label = [], []
                    continue
                char, tag = line.split()
                text.append(char)
                label.append(tag)
        return texts, labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        label_ids = [self.label2id[l] for l in labels] + \
                    [self.label2id["O"]] * (MAX_LEN - len(labels))

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids)
        }

    def __len__(self):
        return len(self.texts)

# ------------------------- 模型结构(BERT + CRF) ------------------------
class BERT_CRF(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)[0]
        emissions = self.classifier(self.dropout(outputs))

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())

# ------------------------- 主程序 ------------------------
def load_labels(files):
    labels = set()
    for f in files:
        with open(f, "r", encoding="utf-8") as data:
            for line in data:
                parts = line.strip().split()
                if len(parts) == 2:
                    labels.add(parts[1])
    labels = sorted(list(labels))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label

def evaluate(model, dataloader, id2label):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            for p, t, mask in zip(outputs, batch["labels"], batch["attention_mask"]):
                p = p[:mask.sum()].tolist()
                t = t[:mask.sum()].tolist()
                preds.extend([id2label[i] for i in p])
                trues.extend([id2label[i] for i in t])
    print(classification_report(trues, preds, digits=4))

def main():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    label2id, id2label = load_labels(["data/train.txt", "data/dev.txt"])
    num_labels = len(label2id)

    train_dataset = POSDataset("data/train.txt", tokenizer, label2id)
    dev_dataset = POSDataset("data/dev.txt", tokenizer, label2id)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    model = BERT_CRF(num_labels).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        logging.info(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

        evaluate(model, dev_loader, id2label)
        model.save_pretrained(f"save_model/pos-bert-crf-epoch{epoch+1}")
        tokenizer.save_pretrained(f"save_model/pos-bert-crf-epoch{epoch+1}")

if __name__ == "__main__":
    print("初始化完成")
    pass # TODO
    # main()
