import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

model = None

def get_model():
    global model
    if model is None:
        logger.info("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model