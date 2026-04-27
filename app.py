import streamlit as st
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="model")

st.title("Sentiment Analysis App")

text = st.text_input("Enter your text")

if text:
    result = classifier(text)

    label = result[0]['label']
    score = result[0]['score']

# Convert label
    if label == "LABEL_1":
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😡"

# Display nicely
    st.write("Sentiment:", sentiment)
    st.write("Confidence:", round(score, 3))