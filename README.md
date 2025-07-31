## Overview

Objective of this project is to develop a [Named Entity Recognition / NER](https://en.wikipedia.org/wiki/Named-entity_recognition) model from the given dataset. The labelled dataset follows the IOB2 tagging scheme which is now the standard tagging scheme. IOB2 is an improvement over original [IOB scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). A list of other common NER tagging schemes has been given below in the appendix section.

Named Entity Recognition involves performing multi-class classification at the token level across a sequence, assigning each token a label that indicates whether it belongs to a named entity and, if so, what type.

## Modelling Approaches for NER

**Traditional ML appraoches**

- Use models such as SVMs or [CRFs / Conditional random fields](https://en.wikipedia.org/wiki/Conditional_random_field) with hand-engineered features (e.g. TF-IDF, POS tags, [gazetteers](https://en.wikipedia.org/wiki/Gazetteer) etc)
- Due to the heavy reliance on manually crafted features they lack of capacity to model sequential dependencies and semantic relationships effectively

**Deep learning Architectures**
 - Architectures like RNNs, CNNs and recently transformer based models like BERT can learn rich contextual representation directly from raw text
 - Local and long range dependencies in a sentence are automatically captured thus enabling more accurate and robust entity recognition without the need for extensive feature engineering

Due to their superior ability, DL models have become the standard approach for modern NER sysmtes and these have been used in this project. 

## Core components in NER

- Tokenization - Convert raw text into smaller units (words, subwords etc)
- Embedding creation - Turn tokens into numerical vectors by using pretrained embeddings (e.g. Word2Vec, GloVe) or Contextual embeddings (e.g. BERT, RoBERTa)
- Sequence Modeling - Use models like BiLSTM, CRF or Transformer to encode context around each token, this captures entity boundaries
- Classification - Predict entity tags e.g `B-LOC`, `I-LOC`, `B-PER` etc
- Post processing - group token level predictions into full entity spans and types

## Experiments

In the initial phase of experimentation, the model architecture comprised an [embeddings layer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html), an [LSTM layer](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM), and a [linear classifier](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) — all of which were trainable. However, with approximately 30,000 unique words in the corpus, learning meaningful embeddings solely from the training data proved impractical.

To address this, pre-trained word embeddings such as Word2Vec or GloVe could be integrated with the LSTM network. However, given the superior performance of Transformer-based architectures in language modeling tasks, the embeddings and LSTM layers were replaced with the [BERT model](https://huggingface.co/google-bert/bert-base-cased).

The final architecture consisted of BERT model followed by a linear classifier layer.

Preliminary experiments using BERT revealed signs of overfitting, indicated by an initial decrease in the loss function followed by a subsequent increase. With over 100 million parameters, fine-tuning the entire BERT model was computationally intensive. As a time-efficient approach for the initial draft, all layers of the BERT model were frozen, and only the final linear classification layer was made trainable.

## Data Processing

A csv file containing training data has been provided. There are around 48,000 sentences. Each sentence has already been split into tokens and each row in the dataset represents a token in a sentence along with its POS & NER tag.

**Splitting the dataset**

We can split the dataset by sentences and each sentence can be represented as a list of word, NER tag tuple. Post this we can split the sentences into train, validation and test set. Currently training set is 60% of total, validation and test are 20% each. 

There are 17 NER tags in the dataset. For simplicity random split was done however there is a possibility of some tags not being present in training/testing split.

Data was converted into `Dataset` class available in the `datasets` [library](https://huggingface.co/docs/datasets/en/index) and saved locally.  

## Model Development

- Training split was used to train the NER model. 
- [Cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) was selected as the loss function.
- The training dataset consisted of 28775 sentences and model was trained for 20 epochs
- The architecture consisted of `bert-base-cased` which generated embeddings for the sentence followed by linear classifier layer to map the embeddings to output labels
- Weights of BERT model were frozen and the linear layer was trained. After the training step, the learned weights of the linear layer were saved to local disk.
- In first iteration, hyperparameter tuning and early stopping has not been used. Thus only the train split was used for model development.

## Model inference & evaluation

- After developing the model, a separate module was developed to make predictions ie perform NER tagging
- This pipeline was run on the validation dataset to understand how the model is performing 
- Predictions were saved as a separate file which can be used to calculate evaluation metrics

**Evaluation**

Evaluation can be done at Token level or Entity Level

*Token level Evaluation*
- Checks if each individual token is assigned the correct label (e.g. B-LOC, I-ORG etc)
- Doesn’t care whether the full entity is correct
- While it is good for debugging model training and is easy to compute, a model can get many tokens right but still fail to extract valid entities

*Entity level Evaluation*
- Stricter but more meaningful in real-world applications
- Libraries available for entity-level NER evaluation
  - seqeval [pypi link](https://pypi.org/project/seqeval/), [Github repo](https://github.com/chakki-works/seqeval)
  - Scrorer class in spaCy ([link](https://spacy.io/api/scorer))
  - Huggingface's evaluate library - [repo link](https://github.com/huggingface/evaluate), [pypi link](https://pypi.org/project/evaluate/), [docs link](https://huggingface.co/docs/evaluate/index) 


## Design Overview

- Modular scripts have been developed for each step of the model development process - data processing, training, inference, evaluation etc
- Backend created using FastAPI app serve the model
- A simple UI has been created (using HTML/CSS/Java script) to manually test the NER model. 

## CICD Pipeline

**Data Ingestion**

While data was initially provided as a CSV file, a production-ready CI/CD pipeline should include automated data ingestion. This can be implemented using AWS Lambda functions to fetch data from sources such as an S3 bucket, a REST API, or a database. This ensures the pipeline can operate continuously and autonomously with real-time or scheduled data updates.

**Data Preprocessing & Storage**

In the current setup, data preprocessing is performed locally, and the processed dataset is stored using Hugging Face’s datasets library. In a production pipeline, processed data should be saved back to a scalable storage solution such as an S3 bucket or a database for downstream accessibility and reproducibility.

**Model Training & Retraining**

Training transformer-based models (e.g., BERT) typically requires GPU acceleration. While initial prototyping and experimentation can be done locally on CPU with a smaller subset of data, full-scale training should be performed on a GPU-enabled instance using AWS SageMaker or an EC2 instance with GPU support. Automating retraining via CI/CD ensures the model stays up-to-date with incoming data.

**Model Storage**

In this project, BERT’s base model weights were kept frozen, and only a lightweight linear classification head was trained. As a result, the trained weights (~<1MB) can be easily stored in the code repository.

However, for production deployments, we may encounter two scenarios:

1. Frozen Base Model: Only the adapter or classifier weights are saved and committed to the repository. At deployment, a script can fetch the base model from external sources such as Hugging Face Hub.

2. Fine-tuned Base Model: If the full model (including base transformer) is fine-tuned, the entire model (which could be several GBs) should be uploaded to a model store (e.g., S3) and downloaded during deployment.

**Containerization**

Docker is the preferred approach for packaging and deploying the model in production. Key considerations include:

1. Image Size & Dependencies: The project relies on heavy libraries such as torch and transformers, which significantly increase the container size. GPU support might require CUDA dependencies, so choosing the right base image is critical—either a minimal Python image or an official PyTorch image.

2. Model Weights: Large model files (especially full transformer models) should not be baked into the container. Instead, a startup script can be included to download model weights from a remote location (e.g., S3) to the target deployment environment.

**Serving the Model**

There are multiple libraries to serve the trained model -

1. [TorchServe](https://docs.pytorch.org/serve/) - This is designed to be easy and flexible tool for serving PyTorch models in production. However the project is no longer actively maintained.

2. FastAPI + Optimized inference - We can use FastAPI to build the end points. Easy to use. We can do the following to reduce latency at runtime 
   1. Load model once on startup (avoid reloading on each request)
   2. Use TorchScript or ONNX: Convert model for faster inference



## APPENDIX

### List of Common NER tagging schemes

| Scheme                     | Description                                        | Notes                                                                                   |
| -------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **IOB** (BIO)              | Inside, Outside, Beginning                         | The original. B for beginning of a chunk, I for inside, O for outside                   |
| **IOB2**                   | Strict BIO variant: every entity starts with **B** | Most widely used. B always starts an entity                                             |
| **IOE / IOE2**             | Inside, Outside, End                               | Like IOB but uses **E** to mark the end of entities                                     |
| **BILOU** (or BIOLU)       | Beginning, Inside, Last, Outside, Unit-length      | More expressive; distinguishes 1-token entities (`U`) and end of multi-token ones (`L`) |
| **BMEWO** (or BMES)        | Begin, Middle, End, Whole, Outside                 | Popular in Asian NLP (esp. Chinese); similar to BILOU                                   |
| **BIOES**                  | Beginning, Inside, Outside, End, Single            | Another name for BILOU, mostly used in academia                                         |
| **Span-based**             | No labels per token; model predicts spans directly | Used in newer transformer models like SpanBERT, etc.                                    |
| **Seq2seq / Pointer**      | Model predicts entity start/end positions like QA  | Used in LLMs, T5-style models, etc.                                                     |
| **CRF-independent binary** | Binary label for each tag (is-entity, is-not)      | Used with classification + post-processing for chunking                                 |
