import torch
from transformers import BertTokenizerFast
from src.functions import NERDataset, BERT_NER

from torch.utils.data import DataLoader

def get_tokenizer(model_name):
    return BertTokenizerFast.from_pretrained(model_name)

def create_dataloader(records, tag2id, tokenizer, batch_size):
    dataset = NERDataset(records, tag2id, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def initialize_model(num_labels):
    model = BERT_NER(num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for param in model.bert.parameters():
        param.requires_grad = False

    return model, device
