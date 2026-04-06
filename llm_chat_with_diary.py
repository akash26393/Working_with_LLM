import time

import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_classic.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from datetime import datetime
import json
import os

# Helper function to load diary data
def load_diary():
    if not os.path.exists("diary.json"):
        # Create empty diary file if it doesn't exist
        with open("diary.json", 'w') as f:
            json.dump({}, f)
        return {}
    with open("diary.json", 'r') as f:
        return json.load(f)

# Helper function to save diary data
def save_diary(data):
    with open("diary.json", 'w') as f:
        json.dump(data, f)

st.title("Chat with Your Diary")

# --- UI: Date & Note Input ---
# Get today's date automatically
today = datetime.now().date()

# Allow the user to pick a date (defaults to today)
selected_date = st.date_input("Select a date", value=today)

with st.form(key='my_form', clear_on_submit=True):
    note = st.text_area(label="Write your diary here.", height=300)
    submit_button = st.form_submit_button(label='Save')

if submit_button:
    if note:
        # Convert date object to string (JSON keys must be strings)
        selected_date = str(selected_date)

        # Load existing diary data
        data = load_diary()

        # Check if an entry already exists for this date
        if selected_date in data.keys():
            # Append new note if it's not a duplicate
            if note not in data[selected_date]:
                data[selected_date] += f"\n\n{note}"
                save_diary(data)
        else:
            # Create a new entry for this date
            data[selected_date] = note
            save_diary(data)

# --- Q&A Section ---
question = st.text_input(label="Enter your question:")
button = st.button("ASK")

if button:
    if question:
        # Initialize LLM
        llm = OllamaLLM(model='qwen3.5:397b-cloud')

        # Load the entire diary (creates file if not exists)
        data = load_diary()

        if not data:
            st.warning("Your diary is empty. Please add some entries first!")
        else:
            # --- RAG Logic using FAISS ---
            # Convert diary entries to Document objects for FAISS
            documents = []
            for date, diary_text in data.items():
                documents.append(Document(
                    page_content=diary_text,
                    metadata={"date": date}
                ))

            # Create FAISS vector store from diary documents
            embeddings = OllamaEmbeddings(model='nomic-embed-text')
            db = FAISS.from_documents(documents, embedding=embeddings)

            # Retrieve top 3 most relevant diary entries
            retriever = db.as_retriever()
            retrieved_docs = retriever.invoke(question)

            # Combine retrieved diary entries as context
            text = ""
            for doc in retrieved_docs:
                date = doc.metadata.get("date", "Unknown")
                # st.markdown(f"📅 Retrieved: {date}")
                text += f"Date: {date}\nDiary: {doc.page_content}\n\n"

            # Define RAG Prompt
            template = """You are a helpful assistant. Please answer the following question based on the below diary:

Question: {question}

Diary: {text}
"""
            prompt_template = PromptTemplate(template=template, input_variables=["text", "question"])

            # Generate Answer
            chain = prompt_template | llm
            start_time = time.perf_counter()
            answer = chain.invoke({"text": text,
                                   "question": question})
            elapsed_seconds = time.perf_counter() - start_time

            st.caption(f"Response time: {elapsed_seconds:.2f} sec")
            st.subheader("Answer:")
            st.markdown(answer)
