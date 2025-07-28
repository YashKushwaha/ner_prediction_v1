import os
import json

from src.config import PROJECT_ROOT, CHECKPOINT_DIR, DATASET_PATH
from src.data_utils import load_dataset, build_records, get_unique_tags, create_tag_mappings
from src.model_utils import get_tokenizer, create_dataloader, initialize_model
from src.train import train_model
from src.inference_pipeline import NERModel

import torch
import sys

# Load and preprocess data
df = load_dataset(DATASET_PATH, split='validation')
print('Num records ->', len(df))

print(df[0])

ner_model = NERModel(checkpoint_dir=CHECKPOINT_DIR)

BATCH_SIZE = 64

def get_batch(iterable, batch_size):

    batch = []
    for record in iterable:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
        
    if batch:
        yield batch 

output = []
counter = 1 
for batch in  get_batch(df, BATCH_SIZE):
    print(counter, end='\t')
    counter+=1
    sentences = [i['tokens'] for i in batch]
    actual_tags = [i['tags'] for i in batch]

    results = ner_model.batch_predict(sentences)


    for sent_result, tags in zip(results,  actual_tags):
        row = []
        for (word, tag), actual_tag in zip(sent_result, tags ):
            #print(f"{word}: {tag} : {actual_tag}")
            row.append([word, tag, actual_tag])
        output.append(row)
  
print()
save_name = os.path.join(DATASET_PATH, 'predictions.json')
print(save_name)
with open(save_name, 'w') as f:
    json.dump(output, f)

