# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Routers
from app.routers import chat            # POST /chat
from app.routers import files, status   # /upload, /admin/reindex, /status

# Single FastAPI app instance
app = FastAPI(title="Pain & Substance-Use AI Agent")

# ---- Static & Templates ----
# Expect these at project root:
#   templates/chat.html, templates/base.html
#   static/app.css, static/app.js
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---- UI Routes ----
@app.get("/chat-ui", response_class=HTMLResponse)
async def chat_ui(request: Request):
    """Web UI for interactive chat (answers + sources, upload + reindex)."""
    return templates.TemplateResponse("chat.html", {"request": request})

# ---- Convenience/help Routes ----
@app.get("/")
def home():
    """Landing info."""
    return {
        "app": "Pain & Substance-Use AI Agent",
        "docs": "/docs",
        "health": "/health",
        "ui": "/chat-ui"
    }

@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}

@app.get("/chat")
def chat_help():
    """Helpful hint if someone hits GET /chat in a browser."""
    return {"detail": "Use POST /chat with JSON: {'thread_id': 'id', 'message': 'your question'}"}

# ---- Include Routers ----
app.include_router(chat.router, prefix="")     # your existing chat pipeline
app.include_router(files.router, prefix="")    # upload & reindex endpoints
app.include_router(status.router, prefix="")   # status endpoint
