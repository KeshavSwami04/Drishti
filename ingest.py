import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="codebase")

def chunk_text(text, filepath, chunk_size=30):
    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i:i+chunk_size])
        chunks.append({
            "text": chunk,
            "start_line": i + 1,
            "filepath": filepath
        })
    return chunks

def delete_file(filename):
    results = collection.get(where={"filepath": filename})
    if results["ids"]:
        collection.delete(ids=results["ids"])

def ingest_file(filename, content):
    delete_file(filename)
    chunks = chunk_text(content, filename)
    for idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk["text"]).tolist()
        collection.add(
            ids=[f"{filename}_{idx}"],
            embeddings=[embedding],
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