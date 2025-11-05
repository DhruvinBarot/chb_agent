#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8000}

echo "== Preflight =="
python -c "import sys; print('Python', sys.version)"
echo "VENV active? -> ${VIRTUAL_ENV:-no}"
test -f .env && echo ".env present" || echo "WARNING: .env not found"

echo "== Server health =="
curl -sS ${BASE_URL}/health | jq .

echo "== Docs available =="
curl -sS -o /dev/null -w "%{http_code}\n" ${BASE_URL}/docs

echo "== POST /chat (no retrieval yet, just routing) =="
curl -sS -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"smoke_t1","message":"Summarize pain and opioid misuse."}' | jq .

echo "== Ingestion check (files exist?) =="
if [ -d data/papers ]; then ls -1 data/papers | wc -l | xargs echo "PDF count:"; else echo "data/papers missing"; fi

echo "== Retrieval-only dry run (if ingested) =="
python - <<'PY'
import os, json
import chromadb
DB_DIR="data/chroma_db"; COLLECTION="papers"
try:
    cl=chromadb.PersistentClient(path=DB_DIR)
    col=cl.get_collection(COLLECTION)
    r=col.query(query_texts=["reciprocal model of pain and substance use"], n_results=2)
    print(json.dumps({"retrieved_n": len(r["ids"][0]) if r and r.get("ids") else 0}, indent=2))
except Exception as e:
    print("Retrieval check skipped/failed:", e)
PY

echo "== POST /chat (expect retrieval + citations if DB populated) =="
curl -sS -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"smoke_t2","message":"Summarize the relationship between chronic pain and substance use."}' | jq .

echo "== Memory check (same thread, two turns) =="
curl -sS -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"mem_t1","message":"Turn 1: hello"}' > /dev/null
curl -sS -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"mem_t1","message":"Turn 2: recall the last thing I said"}' | jq '.context | length'

echo "== Rate-limit check (expect a 429 at some point) =="
for i in {1..7}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/chat" \
    -H "Content-Type: application/json" \
    -d "{\"thread_id\":\"rate_${RANDOM}\",\"message\":\"ping $i\"}")
  echo "Req $i -> $code"
done

echo "== Log file exists? =="
test -f logs/conversation_history.jsonl && echo "logs OK" || echo "logs missing"

echo "== UI route =="
curl -s -o /dev/null -w "%{http_code}\n" ${BASE_URL}/chat-ui

