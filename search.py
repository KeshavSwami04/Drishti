import logging
from typing import List, Dict

import chromadb
from model import get_model

from reranker import rerank

# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# ChromaDB Initialization
# =========================================================

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="codebase"
)

# =========================================================
# Semantic Search Pipeline
# =========================================================

def search(
    query: str,
    n_results: int = 8
) -> List[Dict]:
    """
    Perform semantic retrieval over ingested codebase.

    Workflow:
    1. Embed query
    2. Retrieve top candidates from ChromaDB
    3. Rerank using CrossEncoder
    4. Return best chunks

    Args:
        query (str): User query
        n_results (int): Initial retrieval count

    Returns:
        List[Dict]: Ranked code chunks
    """

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not query.strip():
        logger.warning("Empty query received")
        return []

    try:

        # -------------------------------------------------
        # Query Embedding
        # -------------------------------------------------

        model = get_model()

        query_embedding = model.encode(query).tolist()

        # -------------------------------------------------
        # Vector Search
        # -------------------------------------------------

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # -------------------------------------------------
        # Empty Retrieval Handling
        # -------------------------------------------------

        if not results["documents"][0]:
            logger.warning("No search results found")
            return []

        # -------------------------------------------------
        # Build Chunk Objects
        # -------------------------------------------------

        chunks = []

        for i in range(len(results["documents"][0])):

            chunks.append({
                "text": results["documents"][0][i],
                "filepath": results["metadatas"][0][i]["filepath"],
                "start_line": results["metadatas"][0][i]["start_line"]
            })

        # -------------------------------------------------
        # CrossEncoder Reranking
        # -------------------------------------------------

        try:

            chunks = rerank(query, chunks)

        except Exception as rerank_error:

            logger.exception(
                f"Reranking failed: {str(rerank_error)}"
            )

        logger.info(
            f"Retrieved {len(chunks)} chunks for query"
        )

        return chunks

    except Exception as e:

        logger.exception(
            f"Search pipeline failed: {str(e)}"
        )

        return []


# =========================================================
# Local Testing Entry Point
# =========================================================

if __name__ == "__main__":

    results = search(
        "how does the chat input work"
    )

    for chunk in results:

        print(
            f"File: {chunk['filepath']} "
            f"| Line: {chunk['start_line']}"
        )

        print(chunk["text"])

        print("-" * 60)
