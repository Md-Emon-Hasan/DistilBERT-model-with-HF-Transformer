'''
   Author  : Emon Hasan
   Email   : iconicemon01@gmail.com
   GitHub  : https://github.com/Md-Emon-Hasan
   LinkedIn: https://www.linkedin.com/in/emon-hasan/
   Date    : 01/17/2025
   Time    : 22:13
   Project : Question Answering using distilbert-base-cased-distilled-squad
'''

# Import required libraries
import streamlit as st
from transformers import pipeline

# Load the pre-trained QA pipeline
@st.cache_resource
def load_qa_pipeline():
    # Use a DistilBERT model fine-tuned for QA
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return qa_pipeline

# Initialize the QA pipeline
qa_pipeline = load_qa_pipeline()

# Streamlit UI
st.title("Question Answering with DistilBERT")
st.write(
    "This app uses the `distilbert-base-cased-distilled-squad` model from Hugging Face "
    "to answer questions based on a provided context."
)

# Input text box for context
context = st.text_area("Enter the context (a paragraph or passage):", height=200)

# Input text box for question
question = st.text_input("Enter your question:")

# Perform question answering when both inputs are provided
if context and question:
    with st.spinner("Thinking..."):
        result = qa_pipeline(question=question, context=context)
    st.write("### Answer:")
    st.write(result["answer"])
    st.write("### Confidence Score:")
    st.write(f"{result['score']:.2f}")

# Instructions to run the app
st.markdown("---")
st.markdown("To run the app: `streamlit run qa_app.py`")