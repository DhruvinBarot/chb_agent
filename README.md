# Step 1 — User Input Layer (Backend Skeleton)

This is a minimal FastAPI scaffold for the **User Input** step of your RAG agent roadmap.
It focuses on **accepting chat messages**, **validating**, **routing intent**, **basic safety checks**, and **structured logging**.
Retrieval/RAG/LLM will be attached in later steps.

## Quickstart
```bash
uv venv || python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints
- `POST /chat` → Accepts a user message and thread id, runs intent + safety, and returns a normalized request envelope.
- `GET /health` → Simple health check.

## Extend next
- Wire `/chat` to your retrieval pipeline in Step 3.
- Swap the `basic_safety_check` and `classify_intent` stubs with your real models.
- Add authentication, rate-limits, and observability (Langfuse, OpenTelemetry).
