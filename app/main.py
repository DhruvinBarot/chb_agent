from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.routers import chat

#  Create only one FastAPI instance
app = FastAPI(title="Pain & Substance-Use AI Agent")

#  Jinja2 template directory
templates = Jinja2Templates(directory="templates")

#  Chat UI route
@app.get("/chat-ui", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

#  Root route
@app.get("/")
def home():
    return {
        "app": "Pain & Substance-Use AI Agent",
        "docs": "/docs",
        "health": "/health"
    }

#  Health check
@app.get("/health")
def health():
    return {"status": "ok"}

#  Include routers
app.include_router(chat.router, prefix="")
