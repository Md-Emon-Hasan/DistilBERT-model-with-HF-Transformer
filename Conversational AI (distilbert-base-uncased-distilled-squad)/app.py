'''
   Author  : Emon Hasan
   Email   : iconicemon01@gmail.com
   GitHub  : https://github.com/Md-Emon-Hasan
   LinkedIn: https://www.linkedin.com/in/emon-hasan/
   Date    : 01/18/2025
   Time    : 16:00
   Project : Conversational AI using distilbert-base-uncased-distilled-squad
'''

# Import required libraries
import streamlit as st
from transformers import pipeline

# Set the page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Conversational AI", layout="wide")

# Load DistilBERT QA model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

# Predefined context for the chatbot
context = """
Streamlit is an open-source Python framework that simplifies the development of interactive web applications for data visualization and machine learning. 
It is commonly used by data scientists and machine learning engineers to create dashboards and tools. Streamlit supports multiple data sources, 
including pandas, NumPy, and external APIs. It allows users to deploy models quickly without requiring web development expertise.
"""

# App title
st.title("🤖 Conversational AI with DistilBERT")

# Chat interface
st.write("### Chat with the Assistant")
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:", key="user_input", placeholder="Ask me anything about Streamlit!")

if user_input:
    # Generate response using DistilBERT
    response = qa_pipeline(question=user_input, context=context)
    st.session_state.history.append({"user": user_input, "bot": response['answer']})

# Display chat history
for chat in st.session_state.history[::-1]:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.history = []
    st.experimental_rerun()