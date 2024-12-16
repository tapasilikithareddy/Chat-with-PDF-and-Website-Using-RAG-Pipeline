import streamlit as st
from PyPDF2 import PdfReader
import openai
import os

# Set your API key here
openai.api_key = os.getenv('Your_anthropic_key')  # Replace with your key

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chat_with_pdf(pdf_text, prompt):
    """Use OpenAI API to chat with the PDF's content"""
    response = openai.Completion.create(
        model="gpt-4",  # Use GPT-4 or whichever model you prefer
        prompt=f"Answer the following question based on the content of the PDF:\n{pdf_text}\nQuestion: {prompt}",
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()

# Streamlit app
st.title("Chat with Your PDF")
st.write("Upload a PDF file to chat with its content.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.write("PDF text extracted successfully! Ask your questions below.")

    user_question = st.text_input("Ask a question:")

    if user_question:
        answer = chat_with_pdf(pdf_text, user_question)
        st.write("Answer:", answer)
