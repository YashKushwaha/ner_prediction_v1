import os
from datasets import load_from_disk

def load_dataset(dataset_location, split='train'):
    dataset = load_from_disk(dataset_location)
    return dataset[split]

def build_records(train_df):
    records = [[(tok, tag) for tok, tag in zip(x['tokens'], x['tags'])] for x in train_df]
    return records

def get_unique_tags(train_df):
    unique_tags = set()
    for i in train_df:
        unique_tags.update(i['tags'])
    return unique_tags

def create_tag_mappings(tags):
    tag2id = {tag: i for i, tag in enumerate(sorted(tags))}
    id2tag = {i: tag for tag, i in tag2id.items()}
    return tag2id, id2tag
