import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "data/chroma_db"
COLLECTION = "papers"

def retrieve_relevant_chunks(query: str, n_results: int = 3):
    """Retrieve top matching text chunks with metadata for citation and reasoning."""
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        col = client.get_collection(name=COLLECTION)
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return []

    try:
        results = col.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
    except Exception as e:
        print(f"❌ Query error: {e}")
        return []

    retrieved_chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved_chunks.append({
            "source": meta.get("source"),
            "chunk": meta.get("chunk"),
            "excerpt": doc[:400] + "..." if doc else ""
        })
    return retrieved_chunks
