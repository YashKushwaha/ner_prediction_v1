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