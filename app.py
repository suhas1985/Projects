import streamlit as st
import os
import PyPDF2
import docx
import pptx
import tempfile
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenAI

# Load environment variables
load_dotenv()

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(file):
    text = ""
    with open(file, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from PPTX
def extract_text_from_pptx(file):
    ppt = pptx.Presentation(file)
    text = ""
    for slide in ppt.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

# Streamlit UI
st.title("RAG Chatbot")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "pptx"])

text = ""  # Initialize text variable

if uploaded_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Extract text based on file type
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(tmp_file_path)
    elif uploaded_file.name.endswith(".docx"):
        text = extract_text_from_docx(tmp_file_path)
    elif uploaded_file.name.endswith(".pptx"):
        text = extract_text_from_pptx(tmp_file_path)
    else:
        st.error("Unsupported file type")
        text = ""

    # Clean up temporary file
    os.remove(tmp_file_path)

    if text:
        # GoogleGenAI setup
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found. Please set it in the .env file.")
        else:
            google_genai = GoogleGenAI(api_key=api_key)
            
            # User prompt for querying the document
            user_prompt = st.text_input("Enter your query about the document:")
            if user_prompt:
                response = google_genai.generate_response(
                    prompt=f"{text}\n\nUser: {user_prompt}\nAI:",
                    max_tokens=150
                )
                st.write("Response:")
                st.write(response['choices'][0]['text'].strip())

# Display the extracted text (for debugging purposes, can be removed)
if text:
    st.subheader("Extracted Text")
    st.write(text)
