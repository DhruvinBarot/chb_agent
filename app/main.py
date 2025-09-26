from fastapi import FastAPI
from app.routers import chat

app = FastAPI(title="Pain & Substance-Use Agent — Step 1: User Input")

@app.get("/")
def home():
    return {
        "app": "Pain & Substance-Use Agent — Step 1",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(chat.router, prefix="")
