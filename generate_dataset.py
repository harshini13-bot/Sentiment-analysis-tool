import pandas as pd

positive_reviews = [
    "The product quality exceeded my expectations and works perfectly",
    "Amazing customer service and quick response times",
    "I am very satisfied with this purchase",
    "The application is easy to use and very intuitive",
    "Excellent value for money and highly recommended",
    "The delivery was fast and the packaging was secure",
    "This laptop performs exceptionally well for daily tasks",
    "The food was delicious and the service was outstanding",
    "I love the design and build quality of this product",
    "The hotel staff were friendly and helpful throughout my stay"
]

negative_reviews = [
    "The product stopped working after two days",
    "Terrible customer support and no response to complaints",
    "I regret buying this product",
    "The application crashes frequently and is unusable",
    "Very poor quality and not worth the price",
    "Delivery was delayed and the package arrived damaged",
    "The laptop overheats even during simple tasks",
    "The food was cold and tasted awful",
    "I am extremely disappointed with this purchase",
    "The hotel room was dirty and poorly maintained"
]

data = []

for i in range(100):
    data.append([positive_reviews[i % len(positive_reviews)], "positive"])

for i in range(100):
    data.append([negative_reviews[i % len(negative_reviews)], "negative"])

df = pd.DataFrame(data, columns=["text", "sentiment"])
df.to_csv("reviews.csv", index=False)

print("200-review dataset created successfully!")