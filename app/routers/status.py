from fastapi import APIRouter
from app.services.ingest import collection_stats

router = APIRouter(tags=["status"])

@router.get("/status")
def status():
    return {"ok": True, "chroma": collection_stats()}