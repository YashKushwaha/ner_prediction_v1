import os
import json

from src import config
from src.data_utils import load_dataset, build_records, get_unique_tags, create_tag_mappings
from src.model_utils import get_tokenizer, create_dataloader, initialize_model
from src.train import train_model
import torch

# Load and preprocess data
train_df = load_dataset(config.DATASET_PATH)
print('Num records ->', len(train_df))

records = build_records(train_df)
print("Created records")

unique_tags = get_unique_tags(train_df)
tag2id, id2tag = create_tag_mappings(unique_tags)

# Tokenizer and dataloader
tokenizer = get_tokenizer(config.BERT_MODEL_NAME)
loader = create_dataloader(records, tag2id, tokenizer, config.BATCH_SIZE)

# Model
num_labels = len(unique_tags)
model, device = initialize_model(num_labels)
print('Model moved to ->', device)

# Training
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)

if not os.path.exists(checkpoint_path) or config.OVERWRITE_EXISTING_FILE:
    print('Model training started')
    model = train_model(model, loader, device, num_labels, config.EPOCHS, config.LR)
    torch.save(model.classifier.state_dict(), checkpoint_path)
    print('Model saved')

# Save config
config_data = {
    "num_labels": num_labels,
    "tag2id": tag2id,
    "id2tag": id2tag,
    "max_len": config.MAX_LEN
}

with open(os.path.join(config.CHECKPOINT_DIR, config.CONFIG_FILENAME), "w") as f:
    json.dump(config_data, f)
