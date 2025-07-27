import torch
import json
from transformers import BertTokenizerFast, BertModel

import os
from src.functions import tokenize_for_inference, BERT_NER
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
check_point_save_location = os.path.join(PROJECT_ROOT, 'local_only', 'checkpoints')
config_file = "model_config.json"
config_file = os.path.join(check_point_save_location, config_file)

with open(config_file, "r") as f:
    config = json.load(f)

num_labels = config["num_labels"]
tag2id = config.get("tag2id", None)
id2tag = config.get("id2tag", None)

checkpoint_name = "classifier_weights.pth"
full_name = os.path.join(check_point_save_location, checkpoint_name)

model = BERT_NER(num_labels=num_labels)
model.classifier.load_state_dict(torch.load(full_name))

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tags(sentence, tokenizer, device='cpu'):
    if isinstance(sentence, str):
        words =  word_tokenize(sentence)
    else:
        words = sentence
    features = tokenize_for_inference(words, tokenizer, max_len=128)
    return features

sentence = '''The party is divided over Britain's participation in the Iraq conflict and the continued deployment of 8,500 British troops in that country.'''
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
features = get_tags(sentence, tokenizer, device=device)

model.to(device)
model.eval()

def get_ner_predictions(features, model, device='cpu'):
    input_ids = features["input_ids"].to(device)
    attention_mask = features["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs#.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()
    
    return predictions

predictions = get_ner_predictions(features, model, device)

    # Map predictions back to words
word_ids = features["word_ids"]

final_tags = []
prev_word_idx = None
words =  word_tokenize(sentence)

for idx, word_idx in enumerate(word_ids):
    if word_idx is None or word_idx == prev_word_idx:
        continue
    final_tags.append((words[word_idx], id2tag[str(predictions[idx])]))
    prev_word_idx = word_idx

for i in final_tags:
    print(i)