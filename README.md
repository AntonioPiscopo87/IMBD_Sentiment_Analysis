![how-to-increase-imdb-rating](https://github.com/user-attachments/assets/1fc6f67a-e3f4-4b9f-95e0-2d418b44f41b)

# IMDB Reviews Sentiment Analysis

This repository contains the code and data analysis for performing sentiment analysis on a dataset of IMDB movie reviews. The goal is to classify the sentiment of the reviews as positive or negative using various Natural Language Processing (NLP) techniques and machine learning algorithms.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Sentiment Analysis](#sentiment-analysis)
- [Feature Engineering](#feature-engineering)
- [Models Used](#models-used)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Contributing](#contributing)

## Overview

This project aims to predict the sentiment of movie reviews from the IMDB dataset. Sentiment analysis, a subset of NLP, helps in understanding the emotions conveyed in a text. The dataset used consists of 50,000 movie reviews, categorized as either positive or negative.

## Dataset

The dataset used for this project is the "IMDB Dataset of 50K Movie Reviews," which is freely available on Kaggle. The dataset includes 50,000 reviews, with each review labeled as either positive or negative.

- **Source**: [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Format**: CSV file with two columns: `review` and `sentiment`.

## Requirements

- Python 3.x
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Seaborn
- Matplotlib
- WordCloud

## Installation

To run the code in this repository, you'll need to have Python installed along with the necessary libraries. You can install the required libraries using the following command:

```bash
pip install numpy pandas scikit-learn nltk seaborn matplotlib wordcloud
```

## Data Preprocessing

The preprocessing steps include:
1. **Text Cleaning**: Converting text to lowercase, removing HTML tags, special characters, and digits.
2. **Tokenization**: Breaking down the text into individual words.
3. **Stopwords Removal**: Removing commonly used words that may not contribute to sentiment (e.g., "the", "is", "in").
4. **Lemmatization and Stemming**: Reducing words to their base form.

## Sentiment Analysis

The sentiment analysis is performed using the VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon. VADER is a rule-based model for general sentiment analysis that calculates the polarity scores of the text.

## Feature Engineering

The feature engineering step involves creating numerical representations of the text data using the Count Vectorizer. This step transforms text data into a matrix of token counts, which is used as input for machine learning models.

## Models Used

The following models were used to classify the sentiment of the reviews:
- **Logistic Regression**
- **Gradient Boosting**

These models were selected due to their effectiveness in text classification tasks.

## Evaluation

The performance of the models is evaluated using various metrics such as accuracy, precision, recall, and F1-score. These metrics help in understanding how well the model is performing in classifying the reviews correctly.

## Visualization

Visualizations are created to represent the distribution of sentiments, word clouds for positive and negative reviews, and other exploratory data analysis (EDA) graphs using Seaborn and Matplotlib libraries.

## Contributing

Contributions are welcome! 

---

Feel free to customize this README as per your specific project requirements and add any additional details that you think are necessary.
![image](https://github.com/user-attachments/assets/d9f9399f-4726-41a4-bf6b-bcedc7c415e6)
