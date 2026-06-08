# 😊 Sentiment Analysis Tool

## Overview

The Sentiment Analysis Tool is a Machine Learning-based application that analyzes user reviews and predicts whether the sentiment is **Positive** or **Negative**.

The system uses Natural Language Processing (NLP) techniques such as text preprocessing and TF-IDF vectorization along with a Logistic Regression classifier to perform sentiment classification.

Additionally, a rule-based sentiment detection layer is integrated to improve prediction accuracy for common sentiment expressions.

---

## Features

✅ Text Preprocessing

✅ Sentiment Prediction (Positive / Negative)

✅ TF-IDF Vectorization

✅ Logistic Regression Classification

✅ Rule-Based Sentiment Detection

✅ Confidence Score Display

✅ Interactive Streamlit Interface

---

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-Learn
* TF-IDF Vectorizer
* Logistic Regression
* NLP Preprocessing

---

## Project Structure

sentiment-analysis-tool/

├── app.py

├── train_model.py

├── reviews.csv

├── model.pkl

├── vectorizer.pkl

├── requirements.txt

└── README.md

---

## Workflow

User Review

↓

Text Preprocessing

(Lowercase Conversion & Special Character Removal)

↓

TF-IDF Vectorization

↓

Rule-Based Sentiment Detection

↓

Logistic Regression Model

↓

Sentiment Prediction

↓

Confidence Score Display

---

## Dataset

The dataset contains labeled customer reviews categorized as:

* Positive Reviews
* Negative Reviews

The model is trained using these reviews to learn sentiment patterns.

Dataset Format:

text,sentiment

Amazing customer service and quick response times,positive

Terrible customer support and no response to complaints,negative

---

## Model Training

Run the following command to train the model:

python train_model.py

This generates:

* model.pkl
* vectorizer.pkl

---

## Running the Application

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

---

## Example

Input:

The product quality is excellent and worth the price

Output:

Positive Sentiment 😊

Confidence Score: 92%

Input:

Very poor quality and not worth the money

Output:

Negative Sentiment 😞

Confidence Score: 95%

---

## Future Enhancements

* Multi-class sentiment classification
* Emotion Detection
* Large-scale review datasets
* Deep Learning Models (LSTM/BERT)
* Sentiment Visualization Dashboard

---

## Learning Outcomes

This project helped in understanding:

* Natural Language Processing
* Text Preprocessing
* TF-IDF Feature Extraction
* Machine Learning Classification
* Logistic Regression
* Streamlit Application Development

---

## Author

Harshini Bandla
