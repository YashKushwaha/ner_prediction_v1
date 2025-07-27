import os
import json

from datasets import load_dataset

from transformers import BertTokenizerFast
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from src.functions import tokenize_and_align_labels, NERDataset, tokenize_for_inference, BERT_NER

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import sys

dataset = load_dataset("yash-iitk/my-ner-dataset", split='train')

print("Loaded dataset")

train_df = dataset

records = []
for x in train_df:
    record = [(i,j) for i,j in zip(x['tokens'], x['tags'])]
    records.append(record)

print("Created records")

unique_tags = set()
for i in train_df:
    tags = i['tags']
    unique_tags.update(tags)

tag2id = {tag: i for i, tag in enumerate(sorted(unique_tags))}
id2tag = {i: tag for tag, i in tag2id.items()}

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

dataset = NERDataset(records[:], tag2id, tokenizer)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

num_labels = len(unique_tags)
model = BERT_NER(num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print('model moved to -> ', device)

for param in model.bert.parameters():
    param.requires_grad = False


check_point_save_location = os.path.join(PROJECT_ROOT, 'local_only', 'checkpoints')
os.makedirs(check_point_save_location, exist_ok=True)
checkpoint_name = "classifier_weights.pth"

full_name = os.path.join(check_point_save_location, checkpoint_name)

EPOCHS = 20


def train_model(model, num_labels, loader):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):        
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, num_labels), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    return model

OVERWRITE_EXISTING_FILE = True

if not os.path.exists(full_name) or OVERWRITE_EXISTING_FILE:
    print('Model training started')
    model = train_model(model, num_labels, loader)
    torch.save(model.classifier.state_dict(), full_name)
    print('Model saved')

config = {
    "num_labels": num_labels,
    "tag2id": tag2id,   # optional
    "id2tag": id2tag,   # optional
    "max_len": 128      # optional
}

config_file = "model_config.json"
config_file = os.path.join(check_point_save_location, config_file)

with open(config_file, "w") as f:
    json.dump(config, f)