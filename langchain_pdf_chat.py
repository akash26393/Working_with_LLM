"""
Simple Streamlit app for asking questions about an uploaded PDF using Ollama.

Features:
- Accepts a question from the user.
- Accepts an uploaded PDF file.
- Extracts text from the PDF and combines it with the user question.
- Sends the combined prompt to a single Ollama model and displays the response.
"""
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
import streamlit as st
import hashlib
import time
import fitz

def extract_text_from_pdf(pdf_file):
    # Use PyMuPDF (fitz) for better Gujarati text extraction quality.
    doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
    documents = []
    for i, page in enumerate(doc):
        page_text = page.get_text()
        documents.append(Document(page_content=page_text, metadata={"page": i + 1}))
    doc.close()
    return documents

def split_text_to_chunks(text):
    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    docs = splitter.split_documents(text)
    return docs

def chunks_to_vector_db(chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db

st.title("Chat with PDF!")

with st.form("qa_form", clear_on_submit=True):
    prompt = st.text_area(label= "How can I help you?", key="prompt_input")
    button = st.form_submit_button(label="Enter")

uploaded_file = st.file_uploader("Upload an PDF", type="pdf")

if "qa" not in st.session_state:
    st.session_state.qa = None
if "pdf_key" not in st.session_state:
    st.session_state.pdf_key = None

if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()
    current_pdf_key = hashlib.md5(pdf_bytes).hexdigest()

    # Build the RAG pipeline only once per uploaded PDF.
    if st.session_state.qa is None or st.session_state.pdf_key != current_pdf_key:
        print("First Time....")
        llm = OllamaLLM(model='qwen3-coder:480b-cloud')
        pdf_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text_to_chunks(pdf_text)
        db = chunks_to_vector_db(chunks)
        st.session_state.qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(),
        )
        st.session_state.pdf_key = current_pdf_key

if button and prompt:
    if st.session_state.qa is None:
        st.markdown("Please upload a PDF file")
    else:
        start_time = time.perf_counter()
        result = st.session_state.qa.invoke({"query": prompt})
        elapsed_seconds = time.perf_counter() - start_time
        response = result["result"]
        st.markdown(f"**Question:** {prompt}")
        st.caption(f"Response time: {elapsed_seconds:.2f} sec")
        st.markdown(response)
