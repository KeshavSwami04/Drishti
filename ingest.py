import ast
import logging
from typing import List, Dict

import chromadb
from model import get_model


# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
# =========================================================

CALL_GRAPH_STORE = {}


# =========================================================
# Python AST-Based Chunking
# Splits Python code into logical function/class chunks
# =========================================================

def chunk_python_code(code: str, filepath: str) -> List[Dict]:
    """
    Chunk Python source code using AST nodes.
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
# =========================================================

def extract_calls(code: str, filepath: str) -> List[Dict]:
    """
    Extract internal function-to-function calls.
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

                        if callee in defined_functions:

                            calls.append({
                                "caller": caller,
                                "callee": callee,
                                "filepath": filepath
                            })

    return calls


# =========================================================
# File Deletion
# =========================================================

def delete_file(filename: str):
    """
    Delete all vector entries associated with a file.
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
# =========================================================

def ingest_file(filename: str, content: str) -> int:
    """
    Ingest a file into the vector database.
    """

    logger.info(f"Starting ingestion for {filename}")

    # -----------------------------------------------------
    # Remove old entries before re-ingesting
    # -----------------------------------------------------

    delete_file(filename)

    # -----------------------------------------------------
    # Prevent ingestion of empty files
    # -----------------------------------------------------

    if not content.strip():

        logger.warning(f"{filename} is empty")

        return 0

    # -----------------------------------------------------
    # Python-specific AST chunking
    # -----------------------------------------------------

    if filename.endswith(".py"):

        # ---------------------------------------------
        # Try AST-based semantic chunking first
        # ---------------------------------------------

        chunks = chunk_python_code(
            content,
            filename
        )

        # ---------------------------------------------
        # Fallback:
        # If no functions/classes are found,
        # use generic text chunking instead.
        # ---------------------------------------------

        if not chunks:

            logger.warning(
                f"No AST chunks found in {filename}. "
                "Falling back to text chunking."
            )

            chunks = chunk_text(
                content,
                filename
            )

        # ---------------------------------------------
        # Extract internal function call graph
        # ---------------------------------------------

        calls = extract_calls(
            content,
            filename
        )

        CALL_GRAPH_STORE[filename] = calls

    # -----------------------------------------------------
    # Generic fallback chunking for non-Python files
    # -----------------------------------------------------

    else:

        chunks = chunk_text(
            content,
            filename
        )

    # -----------------------------------------------------
    # Prevent empty chunk insertion
    # -----------------------------------------------------

    if not chunks:

        logger.warning(
            f"No chunks generated for {filename}"
        )

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

    logger.info(
        f"Ingested {len(chunks)} chunks from {filename}"
    )

    return len(chunks)


# =========================================================
# Retrieve All Ingested Files
# =========================================================

def get_ingested_files() -> List[str]:
    """
    Fetch all unique ingested filenames.
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