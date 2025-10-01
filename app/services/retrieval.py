import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "data/chroma_db"
COLLECTION = "papers"

# load Chroma once at startup
client = chromadb.PersistentClient(path=DB_DIR)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    collection = client.get_collection(COLLECTION, embedding_function=embeddings)
except Exception:
    collection = None


def retrieve_relevant_chunks(query: str, n_results: int = 3):
    """Retrieve top-N chunks for a query from Chroma."""
    if collection is None:
        return []

    results = collection.query(query_texts=[query], n_results=n_results)
    docs, metas = results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]

    chunks = []
    for doc, meta in zip(docs, metas):
        chunks.append({
            "source": meta.get("source"),
            "chunk": meta.get("chunk"),
            "content": doc
        })
    return chunks
