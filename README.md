## Overview

Objective of this project is to develop a [Named Entity Recognition / NER](https://en.wikipedia.org/wiki/Named-entity_recognition) model from the given dataset. The labelled dataset follows the IOB2 tagging scheme which is now the standard tagging scheme. IOB2 is an improvement over original [IOB scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)).

Named Entity Recognition involves performing multi-class classification at the token level across a sequence, assigning each token a label that indicates whether it belongs to a named entity and, if so, what type.

## Modelling Approaches for NER

**Traditional ML appraoches**

- Use models such as SVMs or [CRFs](https://en.wikipedia.org/wiki/Conditional_random_field) with hand-engineered features (e.g. TF-IDF, POS tags, [gazetteers](https://en.wikipedia.org/wiki/Gazetteer) etc)
- Due to the heavy reliance on manually crafted features they lack of capacity to model sequential dependencies and semantic relationships effectively

**Deep learning Architectures**
 - Architectures like RNNs, CNNs and recently transformer based models like BERT can learn rich contextual representation directly from raw text
 - Local and long range dependencies in a sentence are automatically captured thus enabling more accurate and robust entity recognition wothout the need for extensive feature engineering

Due to their superior ability, DL models have become the standard approach for modern NER sysmtes and these have been used in this project. 

## Core components in NER

- Tokenization - Convert raw text into smaller units (words, subwords etc)
- Embedding creation - Turn tokens into numerical vectors using: pretrained embeddings (e.g. Word2Vec, GloVe) or Contextual embeddings (e.g. BERT, RoBERTa)
- Sequence Modeling - Use models like BiLSTM, CRF or Transformer to encode context around each token, this captures entity boundaries
- Classification - Predict entity tags e.g `B-LOC`, `I-LOC`, `B-PER` etc
- Post processing - group token level predictions into full entity spans and types

## Experiments

In the initial phase of experimentation, the model architecture comprised an embeddings layer, an LSTM layer, and a linear classifier — all of which were trainable. However, with approximately 30,000 unique words in the corpus, learning meaningful embeddings solely from the training data proved impractical.

To address this, pre-trained word embeddings such as Word2Vec or GloVe could be integrated with the LSTM network. However, given the superior performance of Transformer-based architectures in language modeling tasks, the embeddings and LSTM layers were replaced with the [BERT model](https://huggingface.co/google-bert/bert-base-cased).

Preliminary experiments using BERT revealed signs of overfitting, indicated by an initial decrease in the loss function followed by a subsequent increase. With over 100 million parameters, fine-tuning the entire BERT model was computationally intensive. As a time-efficient approach for the initial draft, all layers of the BERT model were frozen, and only the final linear classification layer was made trainable.

## Data Processing

Given training data contains around 48,000 sentences. Each sentence has already been split into tokens and each row in the dataset represents a token in a sentence along with its POS & NER tag.

**Splitting the dataset**

We can split the dataset by sentences and each sentence can be represented as a list of word, NER tag tuple. Post this we can split the sentences into train, validation and test set. Currently training set is 60% of total, validation and test are 20% each. There are 17 NER tags in the dataset and there is a possibility of some tags not being present in training/testing split. However for simplicity random split was done.

Data was converted into `Dataset` class available in the `datasets` [library](https://huggingface.co/docs/datasets/en/index). And saved locally.  

## Model Development

- Training split was used to train the NER model. 
- [Cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) was selected as the loss function.
- The training dataset consisted of 28775 sentences and model was trained for 20 epochs
- The architecture consisted of `bert-base-cased` which generated embeddings for the sentence followed by linear classifier layer to map the embeddings to output labels
- Weights of BERT model were frozen and the linear layer was trained. After training the learned weights of the linear layer were saved to local disk.
- In first iteration, hyperparameter tuning and early stopping has not been used. Thus only the train split was used for model development.

## Model inference & evaluation

- After developing the model, a separate module was developed to make predictions ie do NER tagging
- This pipeline was run on the validation dataset to udnerstand how the model is performing 
- Predictions were saved as a separate file which can be used to calculate evaluation metrics

**Evaluation**

Evaluation can be done at Token level or Entity Level

Token level Evaluation
- Checks if each individual token is assigned the correct label (e.g. B-LOC, I-ORG etc)
- Doesn’t care whether the full entity is correct
- While it is good for debugging model training and is easy to compute, a model can get many tokens right but still fail to extract valid entities

Entity level Evaluation
- Stricter but more meaningful in real-world applications 


## System Design

- Modular scripts have been developed for each step of the model development process - data processing, training, inference, evaluation etc
- 


