import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data/papers"
DB_DIR = "data/chroma_db"

def load_pdfs():
    """Read all PDFs in the data/papers folder and return (filename, text)."""
    docs = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".pdf"):
            reader = PdfReader(os.path.join(DATA_DIR, fname))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            docs.append((fname, text))
    return docs

def main():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts, metadatas = [], []
    for fname, raw_text in load_pdfs():
        chunks = splitter.split_text(raw_text)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"source": fname, "chunk": i})

    db = Chroma.from_texts(
    texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=DB_DIR,
    collection_name="papers"   
)

    db.persist()
    print(f"✅ Ingested {len(texts)} chunks from {len(metadatas)} PDFs into Chroma at {DB_DIR}")

if __name__ == "__main__":
    main()

