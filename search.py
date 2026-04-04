from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="codebase")

def search(query, n_results=3):
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "filepath": results["metadatas"][0][i]["filepath"],
            "start_line": results["metadatas"][0][i]["start_line"]
        })
    return chunks

if __name__ == "__main__":
    results = search("how does the chat input work")
    for chunk in results:
        print(f"File: {chunk['filepath']} | Line: {chunk['start_line']}")
        print(chunk["text"])
        print("---")