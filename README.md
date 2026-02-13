# Cellula-Week1

 LSTM Toxic Content Classification

This project trains a Bidirectional LSTM (BiLSTM) model to classify textual queries into multiple toxicity categories using deep learning and Natural Language Processing (NLP).

 Project Objective

The goal of this project is to build an LSTM-based model capable of understanding textual context and classifying content into different toxic and non-toxic categories using F1-score as the primary evaluation metric.

 Dataset Description

The dataset contains:

query → user search/query text

image descriptions → textual description of related images

Toxic Category → classification label

Categories:

Safe

Violent Crimes

Suicide & Self-Harm

Elections

Sex-Related Crimes

Child Sexual Exploitation

Non-Violent Crimes

Unknown S-Type

Unsafe

 Preprocessing Pipeline

To improve model performance, the following preprocessing steps were applied:

 Combined query + image descriptions into one feature
 Lowercased all text
 Removed punctuation, numbers, and URLs
 Removed noisy words (e.g., image, photo, shows, description)
 Tokenized text using Keras Tokenizer
 Applied padding to ensure equal sequence length

 Handling Class Imbalance

The dataset suffered from class imbalance, where the Safe class dominated.

To solve this:

Used class weighting during training

Ensured minority classes contributed to learning

Prevented model collapse into predicting only the majority class
