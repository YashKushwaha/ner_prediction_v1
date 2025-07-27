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
    words =  word_tokenize(sentence)
    tokens = tokenize_for_inference(words, tokenizer, 128)
    features = tokenize_for_inference(words, tokenizer, max_len=128)
    



sentence = '''British prime minister Borris Johnson is visiting Paris in December.'''
words =  word_tokenize(sentence)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

tokens = tokenize_for_inference(words, tokenizer, 128)

new_words = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

print('new_words => ', new_words)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device -> ', device)

features = tokenize_for_inference(words, tokenizer, max_len=128)
# Move inputs to the model's device
input_ids = features["input_ids"].to(device)
attention_mask = features["attention_mask"].to(device)
model.to(device)
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs#.logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()

# Map predictions back to words
word_ids = features["word_ids"]

final_tags = []
prev_word_idx = None

for idx, word_idx in enumerate(word_ids):
    if word_idx is None or word_idx == prev_word_idx:
        continue
    try:
        print(words[word_idx], '==',predictions[idx])
    except Exception as e:
        print(word_idx, idx,' ==> ',e)
    final_tags.append((words[word_idx], id2tag[str(predictions[idx])]))
    prev_word_idx = word_idx

print(final_tags)