import streamlit as st
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

# Load your model and tokenizer
model = load_model('my_model.keras')
tokenizer = Tokenizer(num_words=5000)


# Define the clean_review function
def clean_review(text):
    text = text.replace('READ MORE', '')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.strip()
    return text

def predict_sentiment(review):
    review = clean_review(review)
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Streamlit UI
st.title("Sentiment Analysis")
st.write("Enter a review below to analyze its sentiment.")

user_input = st.text_area("Review")

if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"The sentiment of the review is: {sentiment}")
    else:
        st.write("Please enter a review to analyze.")
