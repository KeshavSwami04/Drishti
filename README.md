# Drishti

A codebase question-answering tool that lets you ask plain-English questions about your code and get answers grounded in the actual source, with file and line number citations.

Live demo: https://drishti.streamlit.app

---

## Overview

Reading unfamiliar code is slow. Grepping through files or jumping between your editor and documentation breaks focus. Drishti lets you upload a codebase and ask questions directly, the way you would ask a colleague who has already read it.

This project implements a RAG (Retrieval-Augmented Generation) pipeline. Uploaded files are split into chunks, embedded using a sentence-transformer model, and stored in a local vector database. At query time, the most semantically relevant chunks are retrieved, reranked, and passed as context to a large language model, which generates an answer based solely on your code.

---

## How It Works

1. **Ingestion.** A file is uploaded through the UI. For Python files, the code is parsed with the `ast` module and split at function and class boundaries. Other file types fall back to text-based chunking. Each chunk is embedded using `all-MiniLM-L6-v2` and stored in ChromaDB with its source file and line number as metadata.

2. **Retrieval.** When a question is submitted, it is embedded using the same model. ChromaDB performs an approximate nearest-neighbor search and returns the top 8 most similar chunks by cosine similarity.

3. **Reranking.** The retrieved chunks are passed through a cross-encoder (`ms-marco-MiniLM-L-6-v2`) that scores each chunk against the query more precisely. The top 3 chunks after reranking are used as context.

4. **Generation.** The reranked chunks and the user's question are sent to LLaMA 3.3 70B via the Groq API. The model is instructed to answer only from the provided context and to cite the source file and line number in every response.

---

## Features

- Supports `.py`, `.js`, `.ts`, `.java`, `.cpp`, and `.c` files
- AST-based chunking for Python (preserves function and class boundaries)
- Two-stage retrieval: vector search followed by cross-encoder reranking
- Every answer cites the source file and line number
- Multiple files can be ingested and queried at the same time
- Files can be deleted from the vector store directly from the sidebar

---

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Reranker | sentence-transformers (`ms-marco-MiniLM-L-6-v2`) |
| Vector store | ChromaDB |
| LLM | LLaMA 3.3 70B via Groq API |
| Language | Python |

---

## Running Locally

**1. Clone the repository and set up a virtual environment:**

```bash
git clone https://github.com/KeshavSwami04/Drishti.git
cd Drishti
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

**2. Add your Groq API key:**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

A free API key is available at [console.groq.com](https://console.groq.com).

**3. Run the app:**

```bash
streamlit run app.py
```

---

## Limitations

ChromaDB is currently configured as a local persistent store. On hosted platforms like Streamlit Cloud, the database resets on each server restart, so ingested files are lost between sessions. Replacing ChromaDB with a hosted vector database such as Pinecone or Qdrant would resolve this in a production deployment.

---

## About

Built by Keshav Swami, second-year Electrical Engineering student at IIT Jodhpur.
