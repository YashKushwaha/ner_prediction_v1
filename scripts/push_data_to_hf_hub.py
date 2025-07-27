import os
import huggingface_hub as hf
from datasets import Dataset, DatasetDict



import os
from pathlib import Path
import pandas as pd
import chardet

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

data_file = os.path.join(PROJECT_ROOT, 'local_only', 'ner_dataset.csv')
with open(data_file, 'rb') as f:
    encoding = chardet.detect(f.read())
    print(encoding)

df = pd.read_csv(data_file, encoding=encoding['encoding'], keep_default_na=False, na_values=[])
df['Sentence #']  = df['Sentence #'].replace('', None)
df['Sentence #'] = df['Sentence #'].ffill()

word_tag_dicts = []
for _, sentence_df in df.groupby('Sentence #'):
    tokens = sentence_df['Word'].tolist()
    tags = sentence_df['Tag'].tolist()
    word_tag_dicts.append({"tokens": tokens, "tags": tags})

dataset = Dataset.from_list(word_tag_dicts)

split_ds = dataset.train_test_split(test_size=0.4, seed=42)
test_val = split_ds["test"].train_test_split(test_size=0.5, seed=42)

final_ds = DatasetDict({
    "train": split_ds["train"],
    "validation": test_val["train"],
    "test": test_val["test"]
})

# --- Step 6: Output size info ---
print("Train size:", len(final_ds["train"]))
print("Validation size:", len(final_ds["validation"]))
print("Test size:", len(final_ds["test"]))

hf_token = os.environ['HUGGINGFACE_HUB_TOKEN']
hf.login(token = hf_token)

username = hf.whoami()['name']
final_ds.push_to_hub(f"{username}/my-ner-dataset")