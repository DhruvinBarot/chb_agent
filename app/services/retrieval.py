# app/services/retrieval.py
import chromadb
from typing import List, Dict, Tuple
import re
import os

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_DIR = "data/chroma_db"
COLLECTION = "papers"

MAX_RESULTS = 12
MIN_SCORE = 0.25
TOP_N = 5

_whitespace = re.compile(r"\s+")

def _clean_excerpt(s: str) -> str:
    s = _whitespace.sub(" ", (s or "")).strip()
    return s.replace(" ,", ",").replace(" .", ".")

def _get_embedding_function():
    return SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def passes_relevance(max_score: float, threshold: float = MIN_SCORE) -> bool:
    """Check if the max score passes the relevance threshold."""
    return max_score >= threshold

def retrieve_relevant_chunks(query: str, topic_terms: List[str] = None, n_results: int = TOP_N) -> Tuple[List[Dict], float]:
    """
    Retrieve and filter relevant chunks based on MIN_SCORE threshold.
    
    Args:
        query: The search query
        topic_terms: Optional list of topic terms (currently unused, for future enhancement)
        n_results: Number of top results to return
        
    Returns:
        Tuple of (filtered_items, max_score)
    """
    items, max_score = retrieve(query)
    
    # Filter by minimum score and limit to n_results
    filtered = [item for item in items if item.get("score", 0) >= MIN_SCORE]
    return filtered[:n_results], max_score

def retrieve(query: str) -> Tuple[List[Dict], float]:
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        col = client.get_collection(
            COLLECTION,
            embedding_function=_get_embedding_function()
        )
    except Exception as e:
        print("[RETRIEVAL] get_collection failed:", e)
        return [], 0.0

    res = col.query(
        query_texts=[query],
        n_results=MAX_RESULTS,
        include=["distances", "metadatas", "documents"]
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[1.0] * len(docs)])[0]

    if not docs:
        return [], 0.0

    items: List[Dict] = []
    max_score = 0.0

    for doc, meta, dist in zip(docs, metas, dists):
        score = max(0.0, 1.0 - float(dist))
        max_score = max(max_score, score)
        items.append({
            "source": meta.get("source", "unknown.pdf"),
            "chunk": meta.get("chunk", -1),
            "excerpt": _clean_excerpt(doc)[:1400],
            "score": round(score, 3),
            "distance": round(float(dist), 3)
        })

    items.sort(key=lambda x: x["score"], reverse=True)
    return items, max_score