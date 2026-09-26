# Comment Category Prediction

A multiclass text classification project for predicting comment categories using TF-IDF-based NLP features combined with structured comment-level features.

## Project Overview

This project explores and predicts comment categories from a dataset containing textual comments, voting information, emoticon features, and additional structured attributes.

The workflow includes exploratory data analysis, feature engineering, TF-IDF text representation, model training, validation using Macro F1 score, and final prediction generation for Kaggle submission.

## Dataset

- Training records: 198,000
- Test records: 102,000
- Task: Multiclass comment classification
- Evaluation Metric: Macro F1 Score
- Source: Kaggle Comment Category Prediction Challenge

### Kaggle Competition

[View Kaggle Competition / Dataset](https://www.kaggle.com/competitions/comment-category-prediction-challenge)

> The dataset is not included in this repository. Please use the Kaggle competition link above to access the original data.

## Exploratory Data Analysis

The project includes exploratory analysis of:

- Target label distribution
- Comment length distribution
- Comment length across categories
- Correlation between numerical features
- Upvotes vs. downvotes by label
- Emoticon feature distributions
- Bias-related flag distributions across labels
- Word count distribution
- Missing values and basic dataset statistics

All EDA visualizations and their interpretations are available directly in the project notebook.

## Feature Engineering

The following features were created and used during modeling:

- `comment_length` : character length of each comment
- `vote_diff` : difference between upvotes and downvotes
- `vote_ratio` : upvotes relative to downvotes

### Text Features

TF-IDF was applied to the comment text using:

- Maximum 8,000 features
- Unigrams and bigrams
- Minimum document frequency of 5
- English stop-word removal
- Sublinear TF scaling

The TF-IDF features were combined with structured numerical features including emoticon counts, voting information, comment length, and additional flag attributes.

## Machine Learning Models

Three classification models were evaluated:

1. LightGBM
2. Logistic Regression
3. Multinomial Naive Bayes

A separate TF-IDF + Logistic Regression pipeline was also evaluated as a baseline.

## Model Evaluation

Models were evaluated using a stratified 80/20 train-validation split with **Macro F1 Score**.

| Model | Validation Macro F1 |
|---|---:|
| Logistic Regression Pipeline | 0.6585 |
| LightGBM | 0.8052 |
| Logistic Regression | 0.3353 |
| Multinomial Naive Bayes | 0.4183 |

LightGBM was then retrained using the complete training dataset and used to generate predictions for the Kaggle test set.

## Kaggle Result

**Final Kaggle Macro F1 Score: 0.80936**

The final predictions were converted back to the original category labels and saved as `submission.csv`.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- SciPy
- Matplotlib
- Seaborn

## Repository Contents

```text
comment-category-prediction/
│
├── README.md
├── Comment_category_Prediction_py.ipynb
└── requirements.txt
```

## Notebook

[View Complete Notebook](./Comment_category_Prediction_py.ipynb)
