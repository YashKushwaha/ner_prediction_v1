import torch
from torch import nn
from transformers import BertTokenizerFast, BertModel, PreTrainedTokenizerFast
import torch

from torch.utils.data import Dataset

from src.config import MODEL_DIR

import os
os.environ['TRANSFORMERS_CACHE'] = MODEL_DIR

class NERDataset(Dataset):
    def __init__(self, data, tag2idx, tokenizer, max_len=128):
        self.data = data
        self.tag2idx = tag2idx
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        return tokenize_and_align_labels(sentence, self.tokenizer, self.tag2idx, self.max_len)

class BERT_NER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)  # [batch, seq_len, num_labels]
        return logits


def tokenize_and_align_labels(sentence, tokenizer, tag2idx, max_len):
    words = [w for w, t in sentence]
    labels = [tag2idx[t] for w, t in sentence]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        truncation=True,
        padding='max_length',
        max_length=max_len
    )

    word_ids = encoding.word_ids()
    aligned_labels = []

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(labels[word_idx])
        else:
            aligned_labels.append(-100)
        previous_word_idx = word_idx

    return {
        "input_ids": torch.tensor(encoding["input_ids"]),
        "attention_mask": torch.tensor(encoding["attention_mask"]),
        "labels": torch.tensor(aligned_labels)
    }

def tokenize_for_inference(words, tokenizer, max_len):
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        truncation=True,
        padding='max_length',
        max_length=max_len
    )

    return {
        "input_ids": torch.tensor([encoding["input_ids"]]),
        "attention_mask": torch.tensor([encoding["attention_mask"]]),
        "word_ids": encoding.word_ids()  # useful for decoding output
    }

def tokenize_for_batch_inference(batch_of_word_lists, tokenizer, max_len):

    encoding = tokenizer(
        batch_of_word_lists,
        is_split_into_words=True,
        return_offsets_mapping=True,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )

    # word_ids is not a tensor, it's a list of lists (one per sentence)
    word_ids_batch = [encoding.word_ids(batch_index=i) for i in range(len(batch_of_word_lists))]

    return {
        "input_ids": encoding["input_ids"],  # [batch_size, seq_len]
        "attention_mask": encoding["attention_mask"],
        "word_ids": word_ids_batch  # List[List[int or None]]
    }


