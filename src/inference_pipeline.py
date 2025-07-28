import os
import json
import torch
import nltk

from typing import List, Union
from nltk.tokenize import word_tokenize
from transformers import BertTokenizerFast

from src.functions import tokenize_for_inference, tokenize_for_batch_inference, BERT_NER

nltk.download("punkt")

class NERModel:
    def __init__(self, checkpoint_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self._load_config()
        self._load_model()
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    def _load_config(self):
        config_path = os.path.join(self.checkpoint_dir, "model_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        self.num_labels = config["num_labels"]
        self.tag2id = config["tag2id"]
        self.id2tag = config["id2tag"]

    def _load_model(self):
        self.model = BERT_NER(num_labels=self.num_labels)
        weights_path = os.path.join(self.checkpoint_dir, "classifier_weights.pth")
        self.model.classifier.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentence):
        words = word_tokenize(sentence) if isinstance(sentence, str) else sentence
        
        features = tokenize_for_inference(words, self.tokenizer, max_len=128)

        input_ids = features["input_ids"].to(self.device)
        attention_mask = features["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=2).squeeze().tolist()

        word_ids = features["word_ids"]
        final_tags = self._map_predictions_to_words(predictions, word_ids, words)
        return final_tags

    def batch_predict(self, sentences: List[Union[str, List[str]]]) -> List[List[tuple]]:
        # Tokenize input strings if needed
        batch_word_lists = [
            word_tokenize(s) if isinstance(s, str) else s
            for s in sentences
        ]
        features = tokenize_for_batch_inference(batch_word_lists, self.tokenizer, max_len=128)

        input_ids = features["input_ids"].to(self.device)
        attention_mask = features["attention_mask"].to(self.device)
        word_ids_batch = features["word_ids"]  # List[List[int or None]]

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=2).cpu().tolist()

        batch_results = []
        for sent_idx in range(len(sentences)):
            word_ids = word_ids_batch[sent_idx]
            words = batch_word_lists[sent_idx]
            preds = predictions[sent_idx]

            final_tags = []
            prev_word_idx = None
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx == prev_word_idx:
                    continue
                tag = self.id2tag[str(preds[token_idx])]
                final_tags.append((words[word_idx], tag))
                prev_word_idx = word_idx

            batch_results.append(final_tags)

        return batch_results


    def _map_predictions_to_words(self, predictions, word_ids, words):
        final_tags = []
        prev_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            tag_id = predictions[idx]
            tag = self.id2tag[str(tag_id)]
            final_tags.append((words[word_idx], tag))
            prev_word_idx = word_idx
        return final_tags

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "local_only", "checkpoints")
    ner_model = NERModel(checkpoint_dir=CHECKPOINT_DIR)

    #sentence = """The party is divided over Britain's participation in the Iraq conflict and the continued deployment of 8,500 British troops in that country."""
    sentences = ["""The party is divided over Britain's participation in the Iraq conflict.""",
                 "Iranian officials visited Tokyo this week"                 
    ]

    results = ner_model.batch_predict(sentences)

    for sent_result in results:
        for word, tag in sent_result:
            print(f"{word}: {tag}")
        print("-" * 40)