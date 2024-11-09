import streamlit as st
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np

# Load the tokenizer and model configuration from the saved files
tokenizer = BertTokenizer.from_pretrained(r'C:\Users\DELL\Coding\Email_Classification_Ham_or_Spam\saved_model')
model = TFBertForSequenceClassification.from_pretrained(r'C:\Users\DELL\Coding\Email_Classification_Ham_or_Spam\saved_model')

# Streamlit app UI
st.title('Email Classification (Spam or Ham)')

# Input for email text
email_text = st.text_area("Enter email text:")

if st.button('Classify'):
    if email_text:
        # Tokenize and preprocess the input text
        inputs = tokenizer(email_text, return_tensors="tf", padding=True, truncation=True, max_length=512)
        
        # Predict the label (Spam or Ham)
        predictions = model.predict(inputs['input_ids'])  # Use model for prediction
        predicted_class = np.argmax(predictions.logits, axis=-1)[0]

        # Display the result
        label = "Spam" if predicted_class == 1 else "Ham"
        st.write(f"The email is classified as: {label}")
    else:
        st.write("Please enter an email text.")
