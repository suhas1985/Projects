# app.py
import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2TxtLoader, PPTXLoader
from langchain.embeddings import GeminiEmbedding
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Gemini
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# Streamlit setup
st.title("Multi-Format Document Chatbot with Google Gemini")

# Function to load and process documents
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2TxtLoader(uploaded_file)
        elif uploaded_file.name.endswith(".pptx"):
            loader = PPTXLoader(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            continue
        
        doc = loader.load()
        documents.extend(doc)

    return documents

# Function to build and query vector store
def build_and_query_vector_store(documents):
    embeddings = GeminiEmbedding(model="gemini-pro")  # Use Gemini Embedding
    vectorstore = FAISS.from_documents(documents, embeddings)

    llm = Gemini(model="gemini-pro")  # Use Gemini LLM
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    return qa

# Function to handle user query and display results
def handle_query(qa, query):
    try:
        result = qa.run(query)
        st.write(result)
    except Exception as e:
        st.error(f"Error: {e}")

# Main code execution
uploaded_files = st.file_uploader("Upload PDF, DOCX, or PPTX files", type=["pdf", "docx", "pptx"], accept_multiple_files=True)

if uploaded_files:
    documents = load_documents(uploaded_files)

    if documents:
        # Process and store text for analysis
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Build and query vector store
        qa = build_and_query_vector_store(texts)

        # Get user query and display response
        user_query = st.text_input("Ask a question about the documents:")
        if user_query:
            handle_query(qa, user_query)
