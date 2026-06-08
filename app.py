import streamlit as st
import pickle
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

model = pickle.load(
    open("model.pkl", "rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊"
)

st.title("😊 Sentiment Analysis Tool")

user_text = st.text_area(
    "Enter a review"
)

if st.button("Analyze Sentiment"):

   if st.button("Analyze Sentiment"):

    clean_text = preprocess(user_text)

    transformed_text = vectorizer.transform([clean_text])

    # Rule-based check
    positive_words = [
        "good", "great", "excellent", "best",
        "amazing", "fantastic", "awesome",
        "love", "recommended", "outstanding"
    ]

    negative_words = [
        "bad", "poor", "worst", "terrible",
        "awful", "hate", "disappointed"
    ]

    rule_prediction = None

    for word in positive_words:
        if word in clean_text:
            rule_prediction = "positive"
            break

    for word in negative_words:
        if word in clean_text:
            rule_prediction = "negative"
            break

    # Use rule prediction if found, otherwise ML prediction
    if rule_prediction:
        prediction = rule_prediction
    else:
        prediction = model.predict(transformed_text)[0]

    probabilities = model.predict_proba(transformed_text)[0]

    confidence = round(max(probabilities) * 100, 2)

    if prediction == "positive":

        st.success(
            f"Positive Sentiment 😊"
        )

    else:

        st.error(
            f"Negative Sentiment 😞"
        )

    st.write(
        f"Confidence Score: {confidence}%"
    )

    st.progress(
        int(confidence)
    )