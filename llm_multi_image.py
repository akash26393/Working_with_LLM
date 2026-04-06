"""
File: llm_multi_image.py
Purpose: Streamlit app for chatting with an Ollama vision model using text
and optional multiple uploaded images.

Features:
- Accepts one prompt from the user.
- Supports multiple uploaded images (jpg/png/jpeg).
- Sends prompt + each image to one model and displays each response.
"""

import ollama
import streamlit as st


st.title("Ollama UI using Streamlit")

prompt = st.text_area(label= "How can I help you?")
button = st.button(label= "Enter")

uploaded_files = st.file_uploader("Upload an image", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

if button and prompt:
    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption="Uploaded Image", width="content")
            image_bytes = uploaded_file.getvalue()
            msg = [{"role": "user", "content": prompt, "images": [image_bytes]}]
            response = ollama.chat(model='gemma3:12b-cloud', messages=msg)
            st.markdown(response["message"]["content"])
    else:
        msg = [{"role": "user", "content": prompt}]
        response = ollama.chat(model='gemma3:12b-cloud', messages=msg)
        st.markdown(response["message"]["content"])