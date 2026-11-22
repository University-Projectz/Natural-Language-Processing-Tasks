# twitter-sentiment-analysis-nlp-tasks

A comprehensive Natural Language Processing (NLP) repository for Twitter Sentiment Analysis, covering end-to-end data preprocessing, feature engineering, traditional machine learning baselines, and deep learning sequence architectures.

---

## Repository Structure

```text
twitter-sentiment-analysis-nlp-tasks/
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── processed/
│   │   └── train_cleaned.csv
│   └── raw/
│       ├── test.csv
│       └── train.csv
│
├── models/
│   ├── task-1-linear-svc-tfidf-pipeline.joblib
│   ├── task2_00_deeplearning_keras_tokenizer_vocab20k_maxlen120.pkl
│   ├── task2_01_traditional_svm_linearsvc_balanced_sublinear_tfidf_pipeline.joblib
│   ├── task2_02_traditional_multinomial_naive_bayes_laplace_tfidf_pipeline.joblib
│   ├── task2_03_deeplearning_bidirectional_simple_rnn_maxpooling.keras
│   ├── task2_04_deeplearning_bidirectional_lstm_maxpooling.keras
│   └── task2_05_deeplearning_bidirectional_gru_maxpooling.keras
│
└── Natural-language-processing-tasks/
    ├── nlp-task-1-twitter-sentiment/
    │   ├── nlp-task-1-twitter-sentiment.ipynb
    │   └── README.md
    └── nlp-task-2-twitter-sentiment/
        ├── nlp-task-2-twitter-sentiment.ipynb
        └── README.md
```

---

## Tasks Overview

### Task 1: Classical NLP & Linear SVM Pipeline
- **Notebook**: `Natural-language-processing-tasks/nlp-task-1-twitter-sentiment/nlp-task-1-twitter-sentiment.ipynb`
- **Documentation**: [Task 1 README](Natural-language-processing-tasks/nlp-task-1-twitter-sentiment/README.md)
- **Highlights**:
  - Full text cleaning pipeline: lowercasing, punctuation removal, tokenization, stopword filtering, and lemmatization.
  - TF-IDF feature extraction.
  - Scikit-learn Linear SVM (`LinearSVC`) classifier pipeline.
  - Model serialization and persistence to `models/task-1-linear-svc-tfidf-pipeline.joblib`.

### Task 2: Comparative Study (Traditional ML vs. Deep Learning)
- **Notebook**: `Natural-language-processing-tasks/nlp-task-2-twitter-sentiment/nlp-task-2-twitter-sentiment.ipynb`
- **Documentation**: [Task 2 README](Natural-language-processing-tasks/nlp-task-2-twitter-sentiment/README.md)
- **Highlights**:
  - Fair evaluation on a single stratified 80/20 train/validation split.
  - **Traditional ML**: Linear SVM (balanced weights) and Multinomial Naive Bayes (Laplace smoothing).
  - **Deep Learning**: Keras sequence modeling with Bidirectional Simple RNN, LSTM, and GRU networks featuring Global Max Pooling and Dropout regularization.
  - Structured artifact naming and unified benchmark comparison table.

---

## Requirements & Setup

Install the required dependencies using:

```bash
pip install -r requirements.txt
```
