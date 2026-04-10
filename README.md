# Drishti

Drishti is a code intelligence tool that combines semantic search with static 
analysis to let developers query and explore unfamiliar codebases through 
natural language.

Live demo: https://drishti.streamlit.app
Code: https://github.com/KeshavSwami04/Drishti

---

## What It Does

Upload any code file. Ask questions about it in plain English. Drishti finds 
the relevant functions, answers your question with file and line attribution, 
and can visualize how functions in your codebase call each other.

It does not keyword-match. It understands meaning.

---

## Architecture

The system has three independent layers: ingestion, retrieval, and interface.

**Ingestion**

When a file is uploaded, Drishti uses Python's built-in AST module to parse 
the code rather than splitting it by line count. It walks the abstract syntax 
tree and extracts each function and class definition as its own chunk, 
preserving logical boundaries. This matters because a line-based splitter 
would cut a function in half. An AST-based splitter respects the structure 
of the code.

Simultaneously, it performs static call analysis. For each function 
definition, it walks the AST to find all function calls made inside it. 
It then filters out external library calls by checking against Python's 
builtins, keeping only calls to functions defined within the same file. 
This produces a clean internal call graph without noise from imported 
dependencies.

Chunks are embedded in batch using SentenceTransformers and stored in 
ChromaDB with metadata containing the source file and starting line number.

**Retrieval**

When a query comes in, it is embedded using the same model and used to 
search ChromaDB for the top 8 semantically similar chunks. Those 8 candidates 
are then passed to a CrossEncoder reranker, which scores each chunk against 
the query more precisely than cosine similarity alone can. The top 3 are 
returned as context.

This two-stage retrieval (approximate nearest neighbor followed by 
cross-encoder reranking) is a standard pattern in production search systems. 
The first stage optimizes for recall, the second for precision.

**Interface**

The Streamlit UI handles file upload, chat history, source display, and call 
graph rendering. The call graph is built using NetworkX and rendered with 
Matplotlib. Nodes are color-coded: green for entry points with no callers, 
orange for leaf functions with no callees, and blue for intermediate nodes.

The system prompt constrains the LLM to only answer using provided context 
and to always cite source location, reducing hallucination.

---

## Tech Stack

| Component | Technology |
|---|---|
| Interface | Streamlit |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Reranking | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Vector store | ChromaDB |
| Static analysis | Python AST module |
| Graph | NetworkX + Matplotlib |
| LLM | LLaMA 3.3 70B via Groq API |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/KeshavSwami04/Drishti.git
cd Drishti
```

Create and activate a virtual environment:

```bash
python -m venv env
env\Scripts\activate      # Windows
source env/bin/activate   # Mac/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a .env file with your Groq API key:




---

## Key Design Decisions

**AST chunking over line chunking**
Splitting by line count breaks functions at arbitrary points. Splitting by 
AST node boundaries preserves logical units, which produces more focused 
embeddings and more accurate retrieval.

**Two-stage retrieval**
Cosine similarity on embeddings is fast but imprecise for short queries. 
The CrossEncoder reranker scores each candidate against the full query 
jointly, which significantly improves precision at the cost of a small 
latency increase. Retrieving 8 then reranking to 3 balances recall and 
accuracy.

**Internal call filtering**
Showing all function calls in the graph including library functions like 
print and len produces an unreadable graph. Filtering to only internally 
defined functions makes the visualization meaningful.

**Context length cap**
Retrieved context is capped at 3000 characters before being sent to the 
LLM. This prevents context window overflow on large files while keeping 
the most relevant content.

**Separated system prompt**
The system prompt is defined separately from the user prompt and injected 
as a system role message. This gives the LLM clearer behavioral constraints 
than embedding instructions in the user turn.

---

## Limitations

ChromaDB runs on the local filesystem. On Streamlit Cloud, the filesystem 
resets on server restart, so ingested files do not persist between sessions. 
Replacing ChromaDB with a hosted vector database like Pinecone would fix this.

AST-based call graph only works for Python files. Other languages fall back 
to line-based chunking without call extraction.

The reranker adds latency on large result sets. For production use, this 
should run asynchronously or be cached.

---

## Future Improvements

- Pinecone integration for persistent cloud storage
- GitHub URL ingestion to index entire repositories without manual upload
- RAGAS evaluation framework to measure retrieval quality automatically
- Support for AST parsing in JavaScript and TypeScript
- Streaming LLM responses for faster perceived response time

---
