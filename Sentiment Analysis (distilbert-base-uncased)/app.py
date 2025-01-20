'''
   Author  : Emon Hasan
   Email   : iconicemon01@gmail.com
   GitHub  : https://github.com/Md-Emon-Hasan
   LinkedIn: https://www.linkedin.com/in/emon-hasan/
   Date    : 01/17/2025
   Time    : 18:14
   Project : Sentiment Analysis using distilbert-base-uncased
'''

# Import required libraries
import streamlit as st
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import pipeline

# Load the pre-trained DistilBERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp

# Initialize the model
nlp = load_model()

# Streamlit user interface
st.title("Sentiment Analysis with DistilBERT")

# Text input box
user_input = st.text_area("Enter text to analyze sentiment:")

if user_input:
    # Use the model to analyze sentiment
    result = nlp(user_input)
    
    # Display the results
    st.write("Sentiment Prediction:")
    st.write(f"Label: {result[0]['label']}, with confidence: {result[0]['score']:.2f}")

# Run the app with the command:
# streamlit run app.py