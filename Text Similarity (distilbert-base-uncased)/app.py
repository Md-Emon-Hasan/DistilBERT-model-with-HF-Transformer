'''
   Author  : Emon Hasan
   Email   : iconicemon01@gmail.com
   GitHub  : https://github.com/Md-Emon-Hasan
   LinkedIn: https://www.linkedin.com/in/emon-hasan/
   Date    : 01/17/2025
   Time    : 23:49
   Project : Text Similarity using distilbert-base-uncased
'''

# Import required libraries
import streamlit as st
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Function to get sentence embeddings using DistilBERT
def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Return the embedding of the [CLS] token (first token)
    return outputs.last_hidden_state[:, 0, :].squeeze()

# Load pre-trained DistilBERT model and tokenizer
@st.cache_resource
def load_distilbert_model():
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_distilbert_model()

# Streamlit UI
st.title("Text Similarity using DistilBERT")
st.write(
    "This app calculates the similarity between two sentences using the DistilBERT model."
)

# Input text boxes for two sentences
sentence1 = st.text_area("Enter the first sentence:", height=100)
sentence2 = st.text_area("Enter the second sentence:", height=100)

# Perform text similarity when both sentences are provided
if sentence1 and sentence2:
    with st.spinner("Calculating similarity..."):
        # Get sentence embeddings
        embedding1 = get_sentence_embedding(sentence1, model, tokenizer)
        embedding2 = get_sentence_embedding(sentence2, model, tokenizer)

        # Compute cosine similarity between the two embeddings
        similarity_score = cosine_similarity([embedding1.numpy()], [embedding2.numpy()])[0][0]

    # Display the similarity score
    st.write(f"### Similarity Score: {similarity_score:.4f}")

    # Display similarity level based on threshold
    if similarity_score > 0.8:
        st.write("The sentences are **very similar**.")
    elif similarity_score > 0.5:
        st.write("The sentences are **somewhat similar**.")
    else:
        st.write("The sentences are **not similar**.")

# Instructions to run the app
st.markdown("---")
st.markdown("To run the app: `streamlit run text_similarity_distilbert.py`")
