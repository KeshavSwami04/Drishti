import logging
from typing import List, Dict

from sentence_transformers import CrossEncoder

# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# CrossEncoder Loader
# Lazy loads reranker model only once
# =========================================================

reranker_model = None


def get_reranker():
    """
    Load and cache CrossEncoder reranker model.

    Returns:
        CrossEncoder: Reranker model instance
    """

    global reranker_model

    if reranker_model is None:

        logger.info("Loading reranker model...")

        reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return reranker_model


# =========================================================
# Reranking Pipeline
# =========================================================

def rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 3
) -> List[Dict]:
    """
    Rerank retrieved documents using CrossEncoder.

    Workflow:
    1. Create query-document pairs
    2. Predict relevance scores
    3. Sort by score descending
    4. Return top-k documents

    Args:
        query (str): User query
        docs (List[Dict]): Retrieved chunks
        top_k (int): Number of final chunks

    Returns:
        List[Dict]: Reranked chunks
    """

    # -----------------------------------------------------
    # Input Validation
    # -----------------------------------------------------

    if not query.strip():

        logger.warning("Empty rerank query received")

        return []

    if not docs:

        logger.warning("No documents provided for reranking")

        return []

    try:

        # -------------------------------------------------
        # Load reranker model
        # -------------------------------------------------

        reranker = get_reranker()

        # -------------------------------------------------
        # Create query-document pairs
        # -------------------------------------------------

        pairs = [
            [query, doc["text"]]
            for doc in docs
        ]

        # -------------------------------------------------
        # Predict relevance scores
        # -------------------------------------------------

        scores = reranker.predict(pairs)

        # -------------------------------------------------
        # Combine documents with scores
        # -------------------------------------------------

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # -------------------------------------------------
        # Extract top documents
        # -------------------------------------------------

        top_docs = [
            doc
            for doc, _ in ranked[:top_k]
        ]

        logger.info(
            f"Reranked {len(docs)} documents"
        )

        return top_docs

    except Exception as e:

        logger.exception(
            f"Reranking pipeline failed: {str(e)}"
        )

        # Fallback:
        # Return original docs instead of crashing
        return docs[:top_k]


# =========================================================
# Local Testing Entry Point
# =========================================================

if __name__ == "__main__":

    sample_docs = [
        {"text": "This function handles login."},
        {"text": "This module builds embeddings."},
        {"text": "The graph visualizes calls."}
    ]

    results = rerank(
        "How are embeddings generated?",
        sample_docs
    )

    for r in results:
        print(r["text"])
