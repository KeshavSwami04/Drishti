# Drishti

AI-powered repository intelligence system for semantic code understanding using AST analysis, hybrid retrieval, and retrieval-augmented generation.

<p align="center">
  <a href="https://askdrishti.streamlit.app/">Live Demo</a>
  ·
  <a href="https://github.com/KeshavSwami04/Drishti">Repository</a>
</p>

---

## Overview

Drishti helps developers explore unfamiliar codebases using natural language.

Instead of relying on keyword search, the system performs semantic retrieval over structurally meaningful code chunks extracted through static analysis. Retrieved context is then passed through a retrieval-augmented generation pipeline to generate grounded responses with source attribution.

The project combines:

- Semantic vector retrieval
- AST-based code parsing
- CrossEncoder reranking
- Static call graph analysis
- Retrieval-augmented generation
- Interactive repository exploration

---

# Core Capabilities

- AST-based semantic chunking for Python files
- Hybrid retrieval pipeline with reranking
- Retrieval-augmented code understanding
- Internal function call graph visualization
- Multi-file semantic indexing
- Source-grounded responses with file references
- Lazy-loaded embedding and reranker models
- Interactive repository exploration UI

---

# Why Drishti?

Modern repositories are difficult to navigate due to scale, architectural complexity, and fragmented implementation logic.

Drishti explores how semantic retrieval, static analysis, and retrieval-augmented generation can be combined to improve repository comprehension and developer onboarding.

The system is designed as an engineering-focused exploration of AI-assisted code intelligence rather than a generic chatbot interface.

---

# System Architecture

```text
                ┌──────────────────────┐
                │   Uploaded Files     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  AST/Text Chunking   │
                │ + Call Extraction    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ SentenceTransformer  │
                │   Embedding Model    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │      ChromaDB        │
                │   Vector Database    │
                └──────────┬───────────┘
                           │
                    User Query
                           │
                           ▼
                ┌──────────────────────┐
                │ Semantic Retrieval   │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │ CrossEncoder         │
                │ Reranking            │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │  LLaMA 3.3 via Groq  │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │ Grounded Response    │
                │ + Source Attribution │
                └──────────────────────┘
```

---

# Retrieval Pipeline

## Stage 1: Semantic Retrieval

- Query embeddings generated using `all-MiniLM-L6-v2`
- ChromaDB retrieves semantically similar chunks
- Top-k candidate retrieval optimized for recall

---

## Stage 2: CrossEncoder Reranking

Retrieved chunks are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This improves precision by scoring query-document relevance jointly instead of relying only on embedding similarity.

---

## Stage 3: Grounded Generation

Top-ranked chunks are injected into a constrained LLM prompt using:

```text
LLaMA 3.3 70B via Groq API
```

Responses are grounded in retrieved repository context and include source references.

---

# Static Analysis Engine

The ingestion pipeline performs:

### AST Parsing

Python files are converted into abstract syntax trees using the built-in `ast` module.

---

### Semantic Chunk Extraction

Functions and classes are extracted as independent retrieval units instead of arbitrary fixed-size line windows.

This preserves semantic structure and improves retrieval quality.

---

### Internal Call Extraction

The system statically extracts:

- caller → callee relationships
- internal function dependencies
- repository execution flow

These relationships are used to construct interactive call graph visualizations.

---

### Vector Storage

Embeddings and metadata are stored inside ChromaDB for semantic retrieval.

---

# Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Embedding Model | SentenceTransformers |
| Reranker | CrossEncoder |
| Vector Database | ChromaDB |
| Static Analysis | Python AST |
| Visualization | NetworkX + Matplotlib |
| LLM | LLaMA 3.3 via Groq |
| Language | Python |

---

# Repository Structure

```text
Drishti/
│
├── app.py              # Streamlit frontend and chat pipeline
├── ingest.py           # AST chunking and ingestion pipeline
├── search.py           # Semantic retrieval pipeline
├── reranker.py         # CrossEncoder reranking
├── model.py            # Shared embedding model loader
├── graph.py            # Call graph construction
├── eval.py             # Local retrieval testing
├── requirements.txt
└── README.md
```

---

# Running Locally

## Clone Repository

```bash
git clone https://github.com/KeshavSwami04/Drishti.git

cd Drishti
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv env

env\Scripts\activate
```

### Linux / macOS

```bash
python -m venv env

source env/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Start Application

```bash
streamlit run app.py
```

---

# Example Queries

```text
How does semantic retrieval work?

Explain the ingestion pipeline.

Where is the embedding model initialized?

How are function calls extracted?

Where is reranking implemented?
```

---

# Key Engineering Decisions

## AST-Based Chunking Instead of Fixed Windows

Preserves logical structure and improves retrieval quality compared to arbitrary line-based chunking.

---

## Hybrid Retrieval Architecture

Combines embedding-based recall with CrossEncoder precision.

This improves contextual relevance significantly over pure vector similarity search.

---

## Shared Lazy-Loaded Models

Embedding and reranker models are initialized globally only once to reduce memory overhead and startup latency.

---

## Source-Grounded Responses

The LLM is constrained to answer only using retrieved context, reducing hallucinations and improving reliability.

---

# Current Limitation

The current deployment uses a shared local ChromaDB instance.

As a result, uploaded files are shared across active users on the hosted deployment.

Example:

- User A uploads files
- User B opening the application may see those indexed files

This occurs because the vector database is stored on the shared Streamlit Cloud filesystem.

## Planned Production Fix

Production deployment would isolate users using:

- Session-scoped vector collections
- Hosted vector databases
- Authentication and workspace isolation
- Multi-tenant retrieval architecture

---

# Future Improvements

- Repository-wide GitHub ingestion
- Persistent hosted vector database
- JavaScript and TypeScript AST support
- Streaming LLM responses
- Dockerized deployment
- Repository summarization mode
- Multi-user workspace isolation

---

# Live Application

https://askdrishti.streamlit.app/

---

# Author

Keshav Swami  
Electrical Engineering, IIT Jodhpur

GitHub: https://github.com/KeshavSwami04
