import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text
df = pd.read_csv("reviews.csv")

X = df["text"].apply(preprocess)
y = df["sentiment"]

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()

model.fit(X_vectorized, y)

pickle.dump(
    model,
    open("model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

print("Model trained successfully")