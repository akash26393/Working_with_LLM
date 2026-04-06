import streamlit as st
import ollama

st.title("Ollama UI using Streamlit")

prompt = st.text_area(label= "How can I help you?")
button = st.button(label= "Enter")

if button:
    if prompt:
        response = ollama.generate(model='qwen3-coder:480b-cloud', prompt=prompt)
        # print(response)
        st.markdown(response["response"])