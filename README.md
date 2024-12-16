# RAG Pipeline: Chat with PDF and Website

This repository implements a Retrieval-Augmented Generation (RAG) Pipeline that allows users to interact with semi-structured data extracted from PDF files and websites. The system extracts, chunks, and embeds data from these sources to facilitate efficient similarity-based retrieval and query handling using a pre-trained LLM model.

Project Files
This project includes the following key files:

1. chat_pdf.py
This script enables users to interact with the content of uploaded PDF files using the OpenAI API. It:

Extracts text from PDF files using PyPDF2.
Allows users to upload PDFs via Streamlit.
Processes the PDF text to answer natural language queries.
2. websiterag.py
This script allows users to interact with the content scraped from websites. It:

Crawls and scrapes content from websites using BeautifulSoup.
Segments the scraped text into chunks.
Converts the chunks into vector embeddings using LangChain and SentenceTransformers.
Stores the embeddings in a FAISS index for efficient query-based retrieval.
3. requirements.txt
This file contains all the required dependencies to run the project. To install the dependencies, simply use:

bash
Copy code
pip install -r requirements.txt
