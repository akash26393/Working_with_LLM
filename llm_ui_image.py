"""
Simple Streamlit UI for chatting with an Ollama model.

Features:
- Accepts a text prompt from the user.
- Optionally accepts an uploaded image (jpg/png/jpeg).
- Sends text or text+image to a single Ollama model and displays the response.
"""

import ollama
import streamlit as st


st.title("Ollama UI using Streamlit")

prompt = st.text_area(label= "How can I help you?")
button = st.button(label= "Enter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width="content")
    image_bytes = uploaded_file.getvalue()
    msg = [{"role": "user", "content": prompt, "images": [image_bytes]}]

if button and prompt:
    if not uploaded_file:
        msg = [{"role": "user", "content": prompt}]
    response = ollama.chat(model='gemma3:12b-cloud', messages=msg)
    st.markdown(response["message"]["content"])
