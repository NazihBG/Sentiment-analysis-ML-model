# Sentiment-analysis-ML-model

## Classical Machine Learning, Neural Networks, BERT and LoRA Fine-Tuning

---

## 1. Introduction

This repository presents a **complete end-to-end sentiment analysis project** that progressively explores **multiple generations of Natural Language Processing (NLP) models**.

The project starts with **classical Machine Learning models**, moves through **feature engineering and neural networks**, and culminates with **Transformer-based architectures (BERT)** and **parameter-efficient fine-tuning using LoRA**.

The objective is not only to achieve good performance, but also to:
- Understand the strengths and limitations of each modeling family
- Compare computational cost vs accuracy
- Illustrate modern NLP workflows used in industry and research

This project is suitable for:
- Machine Learning students
- NLP practitioners
- AI recruiters reviewing technical portfolios
- Academic and educational purposes

---

## 2. Problem Statement

The task is a **sentiment classification problem**, where the goal is to predict the sentiment polarity of a given text.

- Input: User-generated text (reviews / comments)
- Output: Sentiment label (e.g. Positive, Negative)

The dataset also contains **additional metadata** (user information, categorical attributes, numerical features), enabling more advanced modeling strategies.

---

## 3. Dataset Description

### 3.1 Data Source
- Dataset obtained from **Kaggle**
- Split into training and test sets

### 3.2 Main Columns
- `text`: Raw text data
- `sentiment_class`: Target label
- `country`: Categorical feature
- `density`: Categorical feature
- `age_of_user`: Numerical feature
- `time_label`: Categorical / temporal feature

---



## 4. Exploratory Data Analysis (EDA)

The exploratory data analysis phase aims to **deeply understand the dataset before any modeling decision**. This step is critical to ensure robustness, detect biases, and guide feature engineering choices.

The following analyses are conducted:

- Sentiment class distribution and imbalance detection
- Text length statistics (mean, median, variance, outliers)
- Vocabulary size and word frequency distribution
- Sentiment distribution across metadata features (`country`, `density`, `time_label`)
- Correlation analysis between numerical features and sentiment labels

Visualizations include:
- Bar plots for sentiment balance
- Boxplots of text length per sentiment
- Heatmaps for feature correlations
- Word clouds for positive and negative classes

This phase directly influences:
- Choice of evaluation metrics
- Sampling strategies
- Model complexity decisions

---

## 5. Text Preprocessing Pipeline

Text preprocessing is **adapted to each modeling family** to ensure optimal performance.

### 5.1 Classical Machine Learning & MLP

For traditional models, aggressive preprocessing is applied:
- Lowercasing
- Removal of URLs, HTML tags, emojis, and punctuation
- Stopword removal
- Lemmatization using spaCy
- Token normalization

This reduces noise and dimensionality in sparse representations.

### 5.2 Transformer-Based Models

For BERT-based models:
- Minimal preprocessing
- Raw text preserved as much as possible
- Subword tokenization using the pretrained BERT tokenizer
- Automatic padding and truncation to a fixed maximum length

### 5.3 Metadata Preprocessing

- One-hot encoding for categorical features
- Standard scaling for numerical features
- Feature alignment for concatenation with text representations

All preprocessing steps are deterministic and reproducible.

---

## 6. Feature Engineering

### 6.1 Classical Text Features

- Bag-of-Words (BoW)
- TF-IDF with unigram and bigram representations
- Vocabulary pruning based on document frequency thresholds

These features are used for classical ML models and MLPs.

### 6.2 Embedding-Based Features

- Pretrained word embeddings (GloVe / FastText)
- Sentence-level representations via pooling (mean, max)
- Normalization of embedding vectors

### 6.3 Hybrid Feature Representations

- Concatenation of text features with structured metadata
- Enables multi-modal sentiment modeling
- Reflects real-world production systems

---

## 7. Classical Machine Learning Models

The following baseline models are implemented:

- Logistic Regression
- Linear Support Vector Machines (SVM)
- Multinomial Naive Bayes
- Random Forest
- Gradient Boosting (XGBoost / LightGBM)

Key characteristics:
- Strong interpretability
- Fast training and inference
- Low hardware requirements
- Reliable baselines for comparison

Hyperparameters are tuned using stratified cross-validation.

---

## 8. Neural Networks – Multi-Layer Perceptron (MLP)

A **fully connected neural network** is trained on engineered features.

Architecture:
- Input layer (TF-IDF + metadata)
- Multiple dense hidden layers
- ReLU activation functions
- Dropout regularization
- Softmax output layer

Training details:
- Adam optimizer
- Early stopping
- Learning rate scheduling
- Batch normalization

This section highlights the transition from classical ML to neural representation learning.

---

## 9. Transformer-Based Model – BERT

### 9.1 Model Selection

- `bert-base-uncased`
- Pretrained on large-scale English corpora

### 9.2 Fine-Tuning Strategy

- Full fine-tuning of all transformer parameters
- Classification head applied to the `[CLS]` token
- Cross-entropy loss function

### 9.3 Training Configuration

- Mixed precision training (FP16)
- Gradient clipping
- Learning rate warm-up
- Linear learning rate decay

This approach yields significant performance improvements at a higher computational cost.

---

## 10. Parameter-Efficient Fine-Tuning with LoRA

### 10.1 Motivation

Fine-tuning large Transformers is expensive. LoRA addresses this by:
- Reducing the number of trainable parameters
- Lowering memory and compute requirements
- Enabling fine-tuning on limited hardware

### 10.2 LoRA Methodology

- Low-rank adapters injected into attention layers
- Base model parameters frozen
- Only adapter matrices are updated during training

### 10.3 Benefits

- Over 90% reduction in trainable parameters
- Comparable performance to full fine-tuning
- Industry-standard approach for adapting large language models

---

## 11. Large Language Models (LLMs)

The project also includes a conceptual and experimental analysis of LLMs:

- Zero-shot sentiment classification
- Few-shot prompting strategies
- Comparison with supervised fine-tuned models

Evaluation dimensions:
- Accuracy
- Latency
- Cost per inference
- Scalability and deployment constraints

---

## 12. Evaluation Metrics

All models are evaluated using consistent metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

Metrics are reported per class and macro-averaged when applicable.

---

## 13. Model Comparison and Analysis

A comprehensive comparison is performed across models based on:

- Predictive performance
- Training time
- Number of parameters
- Inference latency
- Hardware requirements

This analysis supports **data-driven model selection** for real-world deployment.

---

## 14. Reproducibility and Experiment Management

- Fixed random seeds
- Deterministic preprocessing pipelines
- Versioned dependencies
- Modular and configurable training scripts

All experiments are reproducible end-to-end.

---

## 15. Conclusion

This project demonstrates a **complete NLP modeling lifecycle**, from classical machine learning to modern Transformer adaptation techniques such as LoRA.

It emphasizes:
- Methodological rigor
- Engineering best practices
- Clear understanding of performance–efficiency trade-offs

---


