import streamlit as st
import openai
import fitz  # PyMuPDF
import docx
import pptx

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(file) as pdf:
        for page in pdf:
            text += page.get_text()
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

# LangChain implementation
class LangChain:
    def __init__(self, api_key):
        self.api_key = api_key

    def create_chain(self, text):
        return SimpleChain(self.api_key, text)

class SimpleChain:
    def __init__(self, api_key, text):
        self.api_key = api_key
        self.text = text

    def run(self, user_input):
        response = openai.Completion.create(
            engine="davinci-codex",  # or another engine
            prompt=f"{self.text}\n\nUser: {user_input}\nAI:",
            max_tokens=150
        )
        return response.choices[0].text.strip()

# Streamlit UI
st.title("RAG Chatbot")
api_key = st.text_input("Enter your OpenAI API Key", type="password")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "pptx"])

text = ""  # Initialize text variable

if api_key and uploaded_file:
    openai.api_key = api_key

    # Extract text based on file type
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        text = extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith(".pptx"):
        text = extract_text_from_pptx(uploaded_file)
    else:
        st.error("Unsupported file type")
        text = ""

    if text:
        # LangChain setup
        lc = LangChain(api_key=api_key)
        chain = lc.create_chain(text)
        
        # Chatbot interaction
        user_input = st.text_input("Ask something about the document:")
        if user_input:
            response = chain.run(user_input)
            st.write(response)

# Display the extracted text (for debugging purposes, can be removed)
if text:
    st.subheader("Extracted Text")
    st.write(text)
