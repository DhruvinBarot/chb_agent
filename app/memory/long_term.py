import json
import os
from datetime import datetime
from typing import List, Dict

LOG_PATH = "logs/conversation_history.jsonl"
os.makedirs("logs", exist_ok=True)

def store_interaction(thread_id: str, user_message: str, ai_response: str, retrieval: List[Dict]):
    """Append structured interaction to a JSONL log for long-term memory."""
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "thread_id": thread_id,
        "user_message": user_message,
        "ai_response": ai_response,
        "retrieved_docs": [
            {"source": r.get("source"), "chunk": r.get("chunk")} for r in retrieval
        ]
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

def summarize_history(thread_id: str):
    """Return last few exchanges for context reconstruction."""
    if not os.path.exists(LOG_PATH):
        return []
    summaries = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            if rec["thread_id"] == thread_id:
                summaries.append({
                    "role": "user",
                    "message": rec["user_message"]
                })
                summaries.append({
                    "role": "assistant",
                    "message": rec["ai_response"][:150] + "..."
                })
    return summaries[-6:]  # only last 3 exchanges
