# Drishti

A codebase chat tool that lets you ask questions about your code in plain English.

Live demo: https://drishti.streamlit.app

---

## The Problem

Reading unfamiliar code is slow. You either grep through files manually or context-switch between your editor and documentation. Drishti lets you just ask.

---

## How It Works

Upload a code file. Drishti splits it into chunks, converts each chunk into vector embeddings using sentence-transformers, and stores them in ChromaDB. When you ask a question, it embeds your query, retrieves the top 3 most relevant chunks by cosine similarity, and sends them as context to LLaMA 3.3 70B via Groq. The model answers based on your actual code, not general knowledge.

This architecture is called RAG — Retrieval Augmented Generation.

---

## Features

- Upload .py, .js, .ts, .java, .cpp, .c files
- Ask natural language questions about your code
- Every answer includes the source file and line number
- Multiple files can be ingested and queried simultaneously
- Files can be removed from the vector database from the sidebar

---

## Tech Stack

- Streamlit for the UI
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- ChromaDB as the vector store
- Groq API (LLaMA 3.3 70B) for answer generation
- Python

---

## Running Locally

Clone the repo and install dependencies:
```bash
git clone https://github.com/KeshavSwami04/Drishti.git
cd Drishti
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

Create a .env file with your Groq API key:


Free API key at console.groq.com

Run:
```bash
streamlit run app.py
```

---

## Limitations

ChromaDB runs locally, so ingested files reset when the Streamlit Cloud server restarts. A hosted vector database like Pinecone would fix this in production.

---

## About

Built by Keshav Swami, 2nd year Electrical Engineering at IIT Jodhpur.