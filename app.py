import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Transformer Project",
    page_icon="💬",
    layout="centered"
)

# Title
st.title("💬 Sentiment Analysis Transformer Project")
st.subheader("Analyze whether text sentiment is Positive or Negative")

# Load Hugging Face pretrained model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# User input
user_input = st.text_area("Enter your text here:")

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = classifier(user_input)

        label = result[0]['label']
        score = result[0]['score']

        st.success(f"Sentiment: {label}")
        st.info(f"Confidence Score: {score:.2f}")
    else:
        st.warning("Please enter some text.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit + Hugging Face Transformers")