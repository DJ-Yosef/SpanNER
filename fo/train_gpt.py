import os
import torch
import logging
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torchcrf import CRF
from torch import nn
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO)

RUN_TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_NAME = "model/sikubert"
MAX_LEN = 480
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

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
        # 确保标签长度不超过MAX_LEN
        label_ids = [self.label2id[l] for l in labels]
        if len(label_ids) > MAX_LEN:
            label_ids = label_ids[:MAX_LEN]
        else:
            label_ids += [self.label2id["PAD"]] * (MAX_LEN - len(label_ids))

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
            return [torch.tensor(seq) for seq in self.crf.decode(emissions, mask=attention_mask.bool())]

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
    labels.append("PAD")  # 添加'O'标签,表示假标签填充
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
    vailed_labels = set(id2label.values()) - {"PAD", "w"}
    report = classification_report(trues, preds, digits=4, zero_division=0, labels=list(vailed_labels))
    print(report)

    # 记录未预测到的标签
    unpredicted_labels = set(id2label.values()) - set(trues)
    if unpredicted_labels:
        logging.warning(f"未预测到的标签: {unpredicted_labels}")

    # return report["weighted avg"]["f1-score"]

def save_model(model, tokenizer, epoch, run_cnt, model_name=MODEL_NAME):
    time_stamp = RUN_TIME_STAMP
    save_dir = f"save_model/{model_name}_run_{time_stamp}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"run_{run_cnt}_epoch{epoch+1}.pt"))
    tokenizer.save_pretrained(save_dir)
    logging.info(f"模型已保存至 {save_dir}")

def main(train_file="train.txt", dev_file="dev.txt", model_name=MODEL_NAME):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    label2id, id2label = load_labels([f"data/{train_file}", f"data/{dev_file}"])
    num_labels = len(label2id)

    train_dataset = POSDataset(f"data/{train_file}", tokenizer, label2id)
    dev_dataset = POSDataset(f"data/{dev_file}", tokenizer=tokenizer, label2id=label2id)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    model = BERT_CRF(num_labels).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    run_cnt=0

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
        save_model(model, tokenizer, epoch, run_cnt)
        # tokenizer.save_pretrained(f"save_model/pos-bert-crf-epoch{epoch+1}")

if __name__ == "__main__":
    print("初始化完成")
    # pass # TODO
    main("train_all.txt", "dev.txt")
