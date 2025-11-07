# app/routers/files.py
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingest import ingest_all, collection_stats

DATA_DIR = "data/papers"

router = APIRouter(tags=["files"])

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "saved_as": file.filename}

@router.post("/admin/reindex")
def reindex():
    out = ingest_all()
    stats = collection_stats()
    return {"ingest": out, "stats": stats}
