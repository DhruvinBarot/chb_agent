import os, hashlib
from typing import Dict, Tuple
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

DATA_DIR = "data/papers"
DB_DIR = "data/chroma_db"
COLLECTION = "papers"

def _pdf_text(path: str) -> str:
    reader = PdfReader(path)
    txt = []
    for p in reader.pages:
        txt.append((p.extract_text() or "").strip())
    return "\n".join(txt)

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        col = client.get_collection(COLLECTION)
    except:
        col = client.create_collection(COLLECTION)
    return col

def ingest_all() -> Dict:
    """Incremental ingest all PDFs in data/papers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    col = _get_collection()

    # load existing hashes
    existing = {}
    offset=0; page=500
    while True:
        res = col.get(include=["metadatas"], limit=page, offset=offset)
        if not res["ids"]:
            break
        for _id, meta in zip(res["ids"], res["metadatas"]):
            if meta and "sha256" in meta and "doc_id" in meta and meta.get("chunk",-1)==-1:
                existing[meta["doc_id"]] = meta["sha256"]
        offset += page

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    added_docs = 0
    added_chunks = 0

    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".pdf"): 
            continue
        fpath = os.path.join(DATA_DIR, fname)
        sha = _sha256(fpath)
        doc_id = f"doc::{fname}"

        if existing.get(doc_id) == sha:
            continue  # unchanged

        text = _pdf_text(fpath)
        chunks = splitter.split_text(text)

        # delete old records for this doc
        try:
            col.delete(where={"source": fname})
        except Exception:
            pass

        ids, docs, metas = [], [], []
        for i, ch in enumerate(chunks):
            ids.append(f"{doc_id}::chunk::{i}")
            docs.append(ch)
            metas.append({"source": fname, "chunk": i, "sha256": sha, "doc_id": doc_id})

        # tiny manifest row for doc
        ids.append(doc_id)
        docs.append(f"[DOC] {fname}")
        metas.append({"source": fname, "chunk": -1, "sha256": sha, "doc_id": doc_id})

        col.add(ids=ids, documents=docs, metadatas=metas)
        added_docs += 1
        added_chunks += len(chunks)

    return {"added_docs": added_docs, "added_chunks": added_chunks}

def collection_stats() -> Dict:
    """Quick stats for the dashboard."""
    try:
        col = _get_collection()
        res = col.count()
        return {"collection": COLLECTION, "vectors": res, "db_dir": DB_DIR}
    except Exception as e:
        return {"error": str(e), "collection": COLLECTION, "vectors": 0, "db_dir": DB_DIR}