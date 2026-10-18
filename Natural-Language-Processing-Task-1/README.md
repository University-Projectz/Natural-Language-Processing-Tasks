# NLP Lab - Task 1: Basic Text Preprocessing

## Overview
This folder contains the practical implementation for **Task 1** of our Natural Language Processing (NLP) coursework. The objective of this task is to practice the fundamentals of text cleaning, preprocessing, and vectorization. We used the **Disaster Tweets** dataset.

---

## What is covered in this task:
### 1. Missing Value & Text Integrity Checks
- Went beyond surface-level `.isnull()` checks by inspecting raw text entries, empty strings, and null fields across related metadata (`keyword`, `location`).
- Imputed missing attributes systematically to ensure pipeline consistency.

### 2. Conditional Duplicate & Label Conflict Resolution (0 / 1 Labels)
- **The Problem (Contradictory Labels):** Inspecting duplicate instances of identical tweet texts revealed that certain duplicate entries carried contradictory target annotations (the exact same text was marked as `0: Not Disaster` in some rows and `1: Disaster` in others). Dropping duplicates arbitrarily would corrupt the ground truth and inject noise into downstream model training.
- **Conflict Identification:** Grouped duplicate records by `text` and computed the target variance (`var()`). A variance `> 0` immediately flagged active annotation disagreements.
- **Safety-First Resolution Strategy (Favoring False Positives over False Negatives):**
  * In disaster detection systems, failing to identify an actual life-threatening event (**False Negative**) carries severe consequences, including delayed emergency response and risk to human life. Conversely, issuing a false alarm (**False Positive**) causes minor inconvenience but ensures maximum public safety.
  * To reflect this domain reality, whenever annotator agreement was divided (or evenly split at 50/50), we biased the decision boundary toward `target = 1` (Disaster).
  * Calculated the mean target value (`mean()`) per text group:
    * If `mean >= 0.5`, the tweet was systematically unified to `1` (Disaster).
    * If `mean < 0.5`, it was assigned `0` (Not Disaster).
- **Safe Deduplication:** Once labels were reconciled across all conflicting groups, duplicate text rows were safely dropped, retaining clean, unambiguous, and safety-conscious samples.

### 3. Text Preprocessing (NLTK)
- **Normalization:** Converted all tweet text to lowercase.
- **Noise Removal:** Stripped punctuation, special symbols, and redundant whitespaces.
- **Tokenization & Stopwords:** Tokenized sentences into constituent words using `word_tokenize` and eliminated standard English stopwords.
- **Lemmatization:** Applied `WordNetLemmatizer` to reduce tokens to their base lexical roots.

### 4. Vectorization & Sequence Padding (Keras)
- Mapped cleaned tokens into integer indices using Keras `Tokenizer` with an Out-Of-Vocabulary (`<OOV>`) token for unseen words.
- Padded sequences to a fixed cutoff length (`maxlen=16`) based on the dataset distribution, preparing uniform tensors ready for deep learning architectures.

---

## Contents
- `NLP_task1_.ipynb`: The notebook covering the preprocessing steps step-by-step.
- `Dataset/`: Contains the raw CSV data files used for training and testing.

---

## 👥 Contributors
* **Mina Wagih** ([@MinaWagihz](https://github.com/MinaWagihz)).
* **Doha Elemary** ([@dohaelemary](https://github.com/dohaelemary)).
* **Amr Alaa** ([@Amr-Alaa](https://github.com/Amor296)).
