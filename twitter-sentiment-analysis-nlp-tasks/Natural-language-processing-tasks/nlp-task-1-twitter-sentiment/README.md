# NLP Task 1 — Twitter Sentiment Analysis

## Overview
This is an **educational project** I worked on to **practice NLP basics** through a complete Twitter sentiment analysis workflow.

The notebook covers:
- Data loading and cleaning
- Text preprocessing
- Feature extraction
- Model training and evaluation
- Model saving and reloading for inference

## What I implemented

### 1) Setup
- Imported all libraries needed for preprocessing, modeling, evaluation, and model persistence.

### 2) Data loading
- Loaded training and test datasets from project data files.

### 3) Data cleaning
- Checked and handled missing values.
- Removed duplicated rows.
- Resolved duplicate tweets with conflicting labels before training.

### 4) Text preprocessing
Applied standard NLP preprocessing steps:
- Lowercasing text
- Removing punctuation
- Removing extra spaces
- Tokenization (`word_tokenize`)
- Stopword removal
- Lemmatization (`WordNetLemmatizer`)

### 5) Feature engineering
- Converted processed text into numerical features using **TF-IDF**.

### 6) Model training
- Trained a **Linear SVM** classifier using a scikit-learn `Pipeline`:
  - `TfidfVectorizer`
  - `LinearSVC`

### 7) Model evaluation
- Evaluated on validation split using:
  - Accuracy
  - Classification report
  - Confusion matrix visualization

### 8) Model persistence
- Retrained the final model on the full training data.
- Saved model to:
  - `models/task-1-linear-svc-tfidf-pipeline.joblib`
- Reloaded the saved model and ran a sanity-check prediction.

## Key decisions
- Chose **LinearSVC + TF-IDF** as a strong baseline for tweet sentiment classification.
- Kept preprocessing explicit and step-by-step for clarity and reproducibility.

## Output
- A trained and saved sentiment model (`.joblib`)
- Validation metrics and confusion matrix
- Reproducible end-to-end notebook workflow

---

## 👥 Contributors
- Mina Wagih ([@MinaWagihz](https://github.com/MinaWagihz))
- Doha Elemary ([@dohaelemary](https://github.com/dohaelemary))
- Amr Alaa ([@Amr-Alaa](https://github.com/Amor296))

