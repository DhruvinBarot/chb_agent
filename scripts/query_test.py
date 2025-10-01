import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils.embedding_functions import EmbeddingFunction

DB_DIR = "data/chroma_db"
COLLECTION = "papers"

# 🔧 Wrap HuggingFaceEmbeddings so it's Chroma-compatible
class LangchainEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def __call__(self, input: list[str]):
        # Chroma expects List[str] -> List[List[float]]
        return self.model.embed_documents(input)


def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    embedder = LangchainEmbeddingFunction()

    try:
        col = client.get_collection(COLLECTION, embedding_function=embedder)
    except Exception as e:
        print(f"❌ Could not load collection: {e}")
        return

    query = input("Enter your search query: ")
    results = col.query(query_texts=[query], n_results=3)

    print("\n🔎 Top matches:")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\nResult {i+1}:")
        print(f"Source: {meta['source']} | Chunk: {meta['chunk']}")
        print(f"Content: {doc[:300]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
