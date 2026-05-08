import ast
import logging
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# Embedding Model Loader
# Lazy loads the model only once for efficiency
# =========================================================

model = None


def get_model():
    """
    Load and cache the embedding model.

    Returns:
        SentenceTransformer: Embedding model instance
    """

    global model

    if model is None:
        logger.info("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


# =========================================================
# ChromaDB Initialization
# Persistent local vector database storage
# =========================================================

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="codebase"
)


# =========================================================
# Call Graph Storage
# Stores extracted internal function calls per file
#
# Structure:
# {
#     "app.py": [...],
#     "search.py": [...]
# }
# =========================================================

CALL_GRAPH_STORE = {}


# =========================================================
# Python AST-Based Chunking
# Splits Python code into logical function/class chunks
# =========================================================

def chunk_python_code(code: str, filepath: str) -> List[Dict]:
    """
    Chunk Python source code using AST nodes.

    Each function and class becomes an independent chunk.

    Args:
        code (str): Source code content
        filepath (str): Name of source file

    Returns:
        List[Dict]: List of code chunks with metadata
    """

    try:
        tree = ast.parse(code)

    except SyntaxError:
        logger.warning(f"Syntax error while parsing {filepath}")
        return []

    chunks = []

    lines = code.splitlines()

    for node in tree.body:

        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):

            start = node.lineno
            end = node.end_lineno

            chunk = "\n".join(lines[start - 1:end])

            chunks.append({
                "text": chunk,
                "start_line": start,
                "filepath": filepath
            })

    return chunks


# =========================================================
# Generic Text Chunking
# Fallback chunking for non-Python files
# =========================================================

def chunk_text(
    content: str,
    filepath: str,
    chunk_size: int = 40
) -> List[Dict]:
    """
    Split generic text files into fixed-size chunks.

    Args:
        content (str): File content
        filepath (str): Source filename
        chunk_size (int): Number of lines per chunk

    Returns:
        List[Dict]: List of text chunks
    """

    lines = content.splitlines()

    chunks = []

    for i in range(0, len(lines), chunk_size):

        chunk = "\n".join(lines[i:i + chunk_size])

        chunks.append({
            "text": chunk,
            "start_line": i + 1,
            "filepath": filepath
        })

    return chunks


# =========================================================
# Internal Function Call Extraction
# Builds lightweight static call graph using AST
# =========================================================

def extract_calls(code: str, filepath: str) -> List[Dict]:
    """
    Extract internal function-to-function calls.

    Only tracks calls between functions defined
    within the same file.

    Args:
        code (str): Python source code
        filepath (str): Source filename

    Returns:
        List[Dict]: Caller-callee relationships
    """

    try:
        tree = ast.parse(code)

    except SyntaxError:
        logger.warning(f"Could not extract calls from {filepath}")
        return []

    defined_functions = set()
    calls = []

    # -----------------------------------------------------
    # Step 1: Collect all internally defined functions
    # -----------------------------------------------------

    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):
            defined_functions.add(node.name)

    # -----------------------------------------------------
    # Step 2: Extract internal function calls
    # -----------------------------------------------------

    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):

            caller = node.name

            for child in ast.walk(node):

                if isinstance(child, ast.Call):

                    if isinstance(child.func, ast.Name):

                        callee = child.func.id

                        # Only keep internal calls
                        if callee in defined_functions:

                            calls.append({
                                "caller": caller,
                                "callee": callee,
                                "filepath": filepath
                            })

    return calls


# =========================================================
# File Deletion
# Removes file embeddings and graph data
# =========================================================

def delete_file(filename: str):
    """
    Delete all vector entries associated with a file.

    Also clears corresponding call graph data.

    Args:
        filename (str): File to remove
    """

    results = collection.get(
        where={"filepath": filename}
    )

    if results["ids"]:

        collection.delete(
            ids=results["ids"]
        )

        logger.info(f"Deleted existing chunks for {filename}")

    # Remove graph data
    if filename in CALL_GRAPH_STORE:
        del CALL_GRAPH_STORE[filename]


# =========================================================
# Main Ingestion Pipeline
# Handles chunking, embeddings, and vector storage
# =========================================================

def ingest_file(filename: str, content: str) -> int:
    """
    Ingest a file into the vector database.

    Workflow:
    1. Remove existing entries
    2. Chunk source code
    3. Extract call graph (Python only)
    4. Generate embeddings
    5. Store in ChromaDB

    Args:
        filename (str): Name of uploaded file
        content (str): Raw file content

    Returns:
        int: Number of chunks ingested
    """

    logger.info(f"Starting ingestion for {filename}")

    # Remove old entries before re-ingesting
    delete_file(filename)

    # Prevent ingestion of empty files
    if not content.strip():
        logger.warning(f"{filename} is empty")
        return 0

    # -----------------------------------------------------
    # Python-specific AST chunking
    # -----------------------------------------------------

    if filename.endswith(".py"):

        chunks = chunk_python_code(content, filename)

        calls = extract_calls(content, filename)

        CALL_GRAPH_STORE[filename] = calls

    # -----------------------------------------------------
    # Generic fallback chunking
    # -----------------------------------------------------

    else:

        chunks = chunk_text(content, filename)

    # Prevent empty chunk insertion
    if not chunks:

        logger.warning(f"No chunks generated for {filename}")
        return 0

    # -----------------------------------------------------
    # Batch Embedding Generation
    # -----------------------------------------------------

    texts = [c["text"] for c in chunks]

    embeddings = get_model().encode(texts)

    # -----------------------------------------------------
    # Store embeddings in ChromaDB
    # -----------------------------------------------------

    for idx, chunk in enumerate(chunks):

        collection.add(
            ids=[f"{filename}:{idx}"],
            embeddings=[embeddings[idx].tolist()],
            documents=[chunk["text"]],
            metadatas=[{
                "filepath": chunk["filepath"],
                "start_line": chunk["start_line"]
            }]
        )

    logger.info(f"Ingested {len(chunks)} chunks from {filename}")

    return len(chunks)


# =========================================================
# Retrieve All Ingested Files
# =========================================================

def get_ingested_files() -> List[str]:
    """
    Fetch all unique ingested filenames.

    Returns:
        List[str]: List of filenames
    """

    results = collection.get()

    files = set()

    for metadata in results["metadatas"]:
        files.add(metadata["filepath"])

    return list(files)


# =========================================================
# Local Testing Entry Point
# =========================================================

if __name__ == "__main__":

    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    count = ingest_file("app.py", content)

    print(f"Ingested {count} chunks from app.py")
