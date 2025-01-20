'''
   Author  : Emon Hasan
   Email   : iconicemon01@gmail.com
   GitHub  : https://github.com/Md-Emon-Hasan
   LinkedIn: https://www.linkedin.com/in/emon-hasan/
   Date    : 01/17/2025
   Time    : 20:10
   Project : Named Entity Recognition using bert-large-cased-finetuned-conll03-english
'''

# Import required libraries
import streamlit as st
from transformers import pipeline

# Load the pre-trained NER pipeline using a BERT model
@st.cache_resource
def load_ner_pipeline():
    # Use BERT model fine-tuned for NER task
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    return ner_pipeline

# Initialize the NER pipeline
ner_pipeline = load_ner_pipeline()

# Streamlit UI
st.title("Named Entity Recognition with BERT")
st.write(
    "This app uses a BERT-based model fine-tuned for Named Entity Recognition "
    "to identify entities such as persons, organizations, and locations."
)

# Input text box for user input
text_input = st.text_area("Enter the text to extract entities from:", height=200)

# Perform NER when input is provided
if text_input:
    with st.spinner("Processing..."):
        entities = ner_pipeline(text_input)

    # Display the results in a more structured format
    st.write("### Extracted Entities:")
    for entity in entities:
        st.write(f"**Entity**: {entity['word']}")
        st.write(f"**Label**: {entity['entity']}")
        st.write(f"**Confidence Score**: {entity['score']:.2f}")
        st.write("---")

# Instructions to run the app
st.markdown("---")
st.markdown("To run the app: `streamlit run ner_app_bert.py`")
