# NLP Task 2 — Twitter Sentiment Analysis

## Overview
This is an **educational project** I worked on to **compare traditional ML and deep learning approaches** for Twitter sentiment analysis in one reproducible workflow.

The notebook covers:
- Robust data loading from project paths
- One-time stratified train/validation split
- Traditional TF-IDF + classical classifiers
- Deep learning sequence models (RNN/LSTM/GRU)
- Fair model comparison on the same validation set
- Saving all trained artifacts with clear naming

## What I implemented

### 1) Setup
- Imported required libraries for data handling, visualization, classical ML, deep learning, and artifact persistence.
- Set deterministic seeds where needed for reproducibility.

### 2) Data loading
- Loaded the cleaned dataset from project-level paths (with fallback handling).
- Used the agreed dataset location:
  - `data/processed/train_cleaned.csv`

### 3) Data split (once, fair comparison)
- Performed **one stratified 80/20 split** with fixed `random_state`.
- Reused the exact same validation rows for every model family to ensure fair comparison.

### 4) Traditional ML pipeline preparation
- Configured **TF-IDF** features with consistent preprocessing parameters.
- Built scikit-learn pipelines so vectorization + classifier stay coupled and reproducible.

### 5) Deep learning preparation
- Tokenized text using Keras `Tokenizer` (fit on train only to avoid leakage).
- Converted tweets to padded sequences with fixed `MAX_LEN`.
- Computed class weights for imbalance handling.
- Saved tokenizer artifact for future inference.

### 6) Model training (multiple families)
Trained and evaluated the following models:
- **Traditional ML**
  - Linear SVM (LinearSVC)
  - Multinomial Naive Bayes
- **Deep Learning**
  - Bidirectional Simple RNN + pooling
  - Bidirectional LSTM + pooling
  - Bidirectional GRU + pooling

### 7) Evaluation and comparison
- Evaluated all models on the same validation split.
- Compared metrics to identify the strongest approach under consistent conditions.

### 8) Artifact persistence (project-level `models/`)
- Saved all artifacts to project-level:
  - `models/`
- Used explicit, descriptive filenames to make each artifact self-explanatory.

Expected artifact naming pattern includes:
- task id (`task2`)
- model order/index
- family (`traditional` / `deeplearning`)
- algorithm/architecture details

## Key decisions
- Used a single stratified split once for fair side-by-side model comparison.
- Separated traditional and deep learning paths while keeping evaluation protocol consistent.
- Enforced clear, structured artifact naming for easier tracking and grading.
- Anchored paths at project root to avoid notebook working-directory issues.

## Output
- Multiple trained sentiment models (`.joblib` + `.keras`)
- Saved tokenizer (`.pkl`) for deep learning inference
- Reproducible end-to-end notebook workflow for Task 2

---

## 👥 Contributors
- Leo Sabry ([@leleosabry-max](https://github.com/leleosabry-max))
- Mina Wagih ([@MinaWagihz](https://github.com/MinaWagihz))  


