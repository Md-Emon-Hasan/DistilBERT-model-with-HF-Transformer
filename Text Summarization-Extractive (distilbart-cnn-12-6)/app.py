'''
   Author  : Emon Hasan
   Email   : iconicemon01@gmail.com
   GitHub  : https://github.com/Md-Emon-Hasan
   LinkedIn: https://www.linkedin.com/in/emon-hasan/
   Date    : 01/18/2025
   Time    : 20:05
   Project : Text Summarization-Extractive using sshleifer/distilbart-cnn-12-6
'''

# Import required libraries
import streamlit as st
from transformers import pipeline

# Load the pre-trained summarization pipeline using distilBART model
@st.cache_resource
def load_summarization_pipeline():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer

# Initialize the summarizer pipeline
summarizer = load_summarization_pipeline()

# Streamlit UI
st.title("Text Summarization with DistilBART")
st.write(
    "This app uses the `sshleifer/distilbart-cnn-12-6` model for summarizing text. "
    "It is optimized for resource-constrained systems while still providing good-quality summaries."
)

# Input text box for user input
text_input = st.text_area("Enter the text to summarize:", height=300)

# Perform summarization when input is provided
if text_input:
    with st.spinner("Summarizing..."):
        summary = summarizer(text_input, min_length=30, max_length=100)

    # Display the summarized text
    st.write("### Summary:")
    st.write(summary[0]['summary_text'])

# Instructions to run the app
st.markdown("---")
st.markdown("To run the app: `streamlit run app.py`")
