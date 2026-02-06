🧠 Pain & Substance-Use Research AI Agent (RAG)

A domain-aware Retrieval-Augmented Generation (RAG) system that enables interactive querying of peer-reviewed literature on pain, substance use, and behavioral health.
The system retrieves evidence from uploaded PDFs, reasons over the content using LLMs, and returns citation-grounded answers through a web-based chat interface.

✨ Key Features

📄 PDF ingestion & semantic indexing (no fine-tuning required)

🔍 Topic-aware retrieval with reranking (Multi-Query + RRF ready)

🧠 LLM reasoning grounded in retrieved evidence

📚 Automatic citations (paper + chunk level)

💬 Interactive web chat UI (collapsible sources)

🧱 Memory management (short-term + long-term summaries)

🛡️ Safety & domain relevance gating

🚀 Production-ready backend (FastAPI, Docker-friendly)



🧩 High-Level Architecture

User Query
   ↓
Safety & Domain Check
   ↓
Intent Classification + Normalization
   ↓
Topic-Aware Retrieval (ChromaDB)
   ↓
Reranking (Cross-Encoder)
   ↓
LLM Reasoning (RAG Prompt)
   ↓
Answer + Citations
   ↓
Memory Update (Short + Long Term)


📁 Project Structure

.
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── routers/
│   │   ├── chat.py              # /chat API
│   │   ├── status.py            # system status
│   │   └── files.py             # PDF upload
│   ├── services/
│   │   ├── retrieval.py         # Chroma retrieval + rerank
│   │   ├── llm_reasoning.py     # RAG answer generation
│   │   ├── intent.py            # intent classification
│   │   ├── safety.py            # safety filtering
│   │   └── topics.py            # domain & topic logic
│   ├── memory/
│   │   ├── short_term.py        # session context
│   │   └── long_term.py         # summarized history
│   ├── schemas.py               # Pydantic models
│   └── utils/
│       └── rate_limit.py
│
├── scripts/
│   ├── ingest_papers.py         # PDF ingestion pipeline
│   └── query_test.py            # CLI retrieval testing
│
├── data/
│   ├── papers/                  # uploaded PDFs
│   └── chroma_db/               # vector store
│
├── templates/
│   ├── base.html
│   └── chat.html                # web UI
│
├── static/
│   └── styles.css
│
├── requirements.txt
└── README.md

🔄 End-to-End Workflow
1️⃣ Document Ingestion

PDFs are uploaded or placed in data/papers/

Text is extracted, cleaned, chunked

Chunks are embedded and stored in ChromaDB

python scripts/ingest_papers.py

2️⃣ Query Processing

User submits a question via UI or API

Query is normalized and checked for domain relevance

Topic terms guide retrieval

3️⃣ Retrieval & Reranking

Semantic search over embedded chunks

Optional cross-encoder reranking

Low-confidence retrieval is rejected gracefully

4️⃣ LLM Reasoning (RAG)

Retrieved evidence is injected into a structured prompt

LLM generates an answer only using retrieved context

Citations are attached per chunk

5️⃣ Memory Updates

Short-term: conversation window

Long-term: summarized interactions for continuity

6️⃣ Response Delivery

Answer shown first

Sources collapsed & expandable

Clean, citation-backed output

🖥️ Running the App
Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Start Server
uvicorn app.main:app --reload


API Docs → http://127.0.0.1:8000/docs

Chat UI → http://127.0.0.1:8000/chat-ui

📦 Core Dependencies
Backend

FastAPI

Uvicorn

Pydantic v2

Retrieval & Embeddings

ChromaDB

sentence-transformers

CrossEncoder (ms-marco-MiniLM)

LLMs (pluggable)

OpenAI (GPT-4 / GPT-4o)

HuggingFace Inference API

Local models (Ollama / vLLM supported)

Frontend

HTML / CSS

Vanilla JavaScript

Jinja2 templates

🧪 Testing
API Test
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"test1","message":"Summarize pain–opioid misuse mechanisms"}'

Retrieval Sanity Check
python scripts/query_test.py

🛠️ Adding New Papers

Upload PDFs or place them in data/papers/

Re-run ingestion:

python scripts/ingest_papers.py


⚠️ No retraining required — only re-embedding.

🧠 Why RAG (No Fine-Tuning)?

Faster iteration

Lower cost

Full transparency

Always grounded in source documents
