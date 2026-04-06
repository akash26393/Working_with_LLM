"""
Simple Streamlit app for asking questions about an uploaded PDF using Ollama.

Features:
- Accepts a question from the user.
- Accepts an uploaded PDF file.
- Extracts text from the PDF and combines it with the user question.
- Sends the combined prompt to a single Ollama model and displays the response.
"""

import ollama
import streamlit as st
import fitz


def extract_text_from_pdf(uploaded_file):
    # fitz.open() opens the PDF document.
    # It can handle file streams directly from Streamlit in many cases.
    doc = fitz.open(uploaded_file)
    text = ""

    # Loop through every page in the document
    for page in doc:
        # .get_text() extracts the plain text content from the page.
        text += page.get_text()

    return text

st.title("Ollama UI using Streamlit")

prompt = st.text_area(label= "How can I help you?")
button = st.button(label= "Enter")

uploaded_file = st.file_uploader("Upload an PDF", type="pdf")
if button and prompt:
    if not uploaded_file:
        st.markdown("Please upload a PDF file")
    else:
        pdf_text = extract_text_from_pdf(uploaded_file)
        combined_prompt = f"Based on the following content: {pdf_text}\n\nQuestion: {prompt}"
        response = ollama.generate(model='qwen3-coder:480b-cloud', prompt=combined_prompt)
        st.markdown(response["response"])
