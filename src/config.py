import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'local_only', 'my-ner-dataset-local')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'local_only', 'checkpoints')
CHECKPOINT_NAME = "classifier_weights.pth"
CONFIG_FILENAME = "model_config.json"
MAX_LEN = 128
BERT_MODEL_NAME = "bert-base-cased"

MODEL_DIR = CHECKPOINT_DIR

EPOCHS = 20
BATCH_SIZE = 64
LR = 5e-4
OVERWRITE_EXISTING_FILE = False
