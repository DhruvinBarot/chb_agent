🧠 Pain & Substance-Use AI Agent

This project is an AI-powered assistant for research in pain, substance use, and health behaviors.
It ingests academic papers (PDFs), builds semantic embeddings, and allows you to query for relevant passages using natural language.

We are currently at Step 3 of the roadmap: ingestion + retrieval.

📂 Project Structure
step1_user_input_backend/
├─ app/                     # FastAPI app
│  ├─ main.py
│  ├─ routers/chat.py
│  ├─ services/
│  │   ├─ intent.py
│  │   ├─ safety.py
│  │   └─ retrieval.py
│  ├─ utils/rate_limit.py
│  └─ schemas.py
├─ data/
│  ├─ papers/               # place your PDFs here
│  └─ chroma_db/            # auto-generated Chroma vector DB
├─ scripts/
│  ├─ ingest_papers.py      # build embeddings from PDFs
│  └─ query_test.py         # test queries against DB
├─ .gitignore
├─ requirements.txt
└─ README.md

⚙️ Setup
1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-pain-substance-agent.git
cd ai-pain-substance-agent

2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


(If you don’t have requirements.txt, install manually:)

pip install fastapi uvicorn chromadb langchain-huggingface langchain-community sentence-transformers PyPDF2

📥 Step 3.1 – Ingest PDFs

Place your research papers in:

data/papers/


Run the ingestion script:

python scripts/ingest_papers.py


You should see:

✅ Ingested XX chunks from YY PDFs into Chroma at data/chroma_db

🔎 Step 3.2 – Test Retrieval

Run the query test script:

python scripts/query_test.py


You’ll be prompted for a query:

Enter your search query: opioid misuse behavioral impacts


Example output:

🔎 Top matches:

Result 1:
Source: paper1.pdf | Chunk: 12
Content: Opioid misuse is associated with...
------------------------------------------------------------
Result 2:
Source: paper2.pdf | Chunk: 4
Content: Behavioral consequences include...
------------------------------------------------------------

🌐 Step 2 + 3 Integration (API)

You can also run the FastAPI app to use /chat:

uvicorn app.main:app --reload


Open http://127.0.0.1:8000/docs
 → Try the POST /chat endpoint with:

{
  "thread_id": "t1",
  "message": "summarize behavioral impacts of opioid misuse"
}


Response includes retrieval with the top matches.

🚀 Roadmap

✅ Step 1: User Input (FastAPI endpoint)

✅ Step 2: Memory Layer (short-term session memory)

✅ Step 3: Ingestion + Retrieval (PDFs → ChromaDB)

🔜 Step 4: LLM Processing (augment queries with retrieved passages)

🔜 Step 5: Interactive Chatbot Interface

🛑 .gitignore

We keep large/generated files out of Git:

.venv/

data/chroma_db/

__pycache__/