import streamlit as st
import pickle

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

st.markdown("""
### Features

✅ TF-IDF Vectorization

✅ Naive Bayes Classifier

✅ Positive / Negative Prediction

✅ Interactive UI
""")

user_text = st.text_area(
    "Enter a review"
)

if st.button("Analyze Sentiment"):

    transformed_text = vectorizer.transform(
        [user_text]
    )

    prediction = model.predict(
        transformed_text
    )[0]

    if prediction == "positive":

        st.success(
            "Positive Sentiment 😊"
        )

    else:

        st.error(
            "Negative Sentiment 😞"
        )