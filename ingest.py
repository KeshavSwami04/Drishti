import chromadb
from sentence_transformers import SentenceTransformer
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="codebase")

import ast
ALL_CALLS = []

def chunk_python_code(code, filepath):
    tree = ast.parse(code)
    chunks = []

    lines = code.splitlines()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = node.lineno
            end = node.end_lineno

            chunk = "\n".join(lines[start-1:end])

            chunks.append({
                "text": chunk,
                "start_line": start,
                "filepath": filepath
            })

    return chunks
def extract_calls(code, filepath):
    tree = ast.parse(code)

    defined_functions = set()
    calls = []

    # 🔹 Step 1: collect defined functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined_functions.add(node.name)

    # 🔹 Step 2: collect calls ONLY if internal
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

def delete_file(filename):
    results = collection.get(where={"filepath": filename})
    if results["ids"]:
        collection.delete(ids=results["ids"])

        import ast

def chunk_python_code(code, filepath):
    tree = ast.parse(code)
    chunks = []

    lines = code.splitlines()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = node.lineno
            end = node.end_lineno

            chunk = "\n".join(lines[start-1:end])

            chunks.append({
                "text": chunk,
                "start_line": start,
                "filepath": filepath
            })

    return chunks

def ingest_file(filename, content):
    delete_file(filename)

    # 🔹 Choose chunking strategy
    if filename.endswith(".py"):
        chunks = chunk_python_code(content, filename)

        # 🔥 NEW: extract function calls
        global ALL_CALLS
        calls = extract_calls(content, filename)
        ALL_CALLS.extend(calls)

    else:
        chunks = chunk_text(content, filename)

    # 🔹 Batch embedding (efficient)
    texts = [c["text"] for c in chunks]
    embeddings = get_model().encode(texts)

    # 🔹 Store in Chroma
    for idx, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{filename}_{idx}"],
            embeddings=[embeddings[idx].tolist()],
            documents=[chunk["text"]],
            metadatas=[{
                "filepath": chunk["filepath"],
                "start_line": chunk["start_line"]
            }]
        )

    return len(chunks)
def get_ingested_files():
    results = collection.get()
    files = set()
    for metadata in results["metadatas"]:
        files.add(metadata["filepath"])
    return list(files)

if __name__ == "__main__":
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    count = ingest_file("app.py", content)
    print(f"Ingested {count} chunks from app.py")