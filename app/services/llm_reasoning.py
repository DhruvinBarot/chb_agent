# app/services/llm_reasoning.py
from typing import List, Dict, Tuple
import os
import re

# Optional: lightweight cleaning for OCR-y snippets
_ws = re.compile(r"\s+")

def _clean(s: str) -> str:
    return _ws.sub(" ", (s or "")).strip()

def _build_context(retrieved: List[Dict]) -> str:
    """Join retrieved snippets into a single CONTEXT block."""
    parts = []
    for i, r in enumerate(retrieved, 1):
        src = r.get("source", "unknown.pdf")
        chk = r.get("chunk", -1)
        txt = _clean(r.get("excerpt", ""))
        parts.append(f"[{i}] {src} (chunk {chk}): {txt}")
    return "\n".join(parts)

def _build_citations(retrieved: List[Dict]) -> List[str]:
    cites = []
    seen = set()
    for r in retrieved:
        key = (r.get("source"), r.get("chunk"))
        if key in seen:
            continue
        seen.add(key)
        cites.append(f"{r.get('source','unknown.pdf')} (chunk {r.get('chunk',-1)})")
    return cites

def _fallback_answer(user_query: str, retrieved: List[Dict]) -> str:
    """If no LLM configured, synthesize a concise extractive answer from the top chunks."""
    if not retrieved:
        return ("I don’t have sufficient in-corpus evidence to answer that. "
                "Please upload relevant papers or re-run ingestion.")
    bullets = []
    for r in retrieved[:3]:
        src = r.get("source", "unknown.pdf")
        chunk = r.get("chunk", -1)
        txt = _clean(r.get("excerpt", ""))[:400]
        bullets.append(f"- From {src} (chunk {chunk}): {txt}")
    return ("Answer (from retrieved context only):\n"
            + "\n".join(bullets)
            + "\n\n(Use Upload/Reindex to add more sources or refine your query.)")

def generate_answer(
    user_query: str,
    retrieved: List[Dict],
    domain_ok: bool,
    relevant: bool
) -> Tuple[str, List[str]]:
    """
    Returns (answer_text, citations). This signature matches your router call.

    Behavior:
      - If out of domain: refuse politely.
      - If weak/empty retrieval: say insufficient evidence.
      - Else: Try LLM (if configured); otherwise fallback to extractive summary.
    """
    # 0) Out-of-domain guard
    if not domain_ok:
        return (
            "This question appears outside the agent’s scope (pain, substance use, and related behavioral health). "
            "Please rephrase within scope or upload relevant PDFs.",
            []
        )

    # 1) Evidence check
    if not relevant or not retrieved:
        return (
            "I don’t have enough in-corpus evidence to answer this confidently. "
            "Consider uploading opioid/pain/substance-use papers relevant to your question, then reindex.",
            []
        )

    # 2) Prepare context and citations
    context_text = _build_context(retrieved)
    citations = _build_citations(retrieved)

    # 3) Try a Hugging Face inference client if available; otherwise fallback
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_API_TOKEN")
    HF_MODEL = os.getenv("HF_TEXTGEN_MODEL", "HuggingFaceH4/zephyr-7b-beta")

    system = (
        "You are a careful research assistant for pain/substance-use. "
        "Use ONLY the provided CONTEXT. If it’s insufficient, say so. "
        "Structure the answer concisely (2–5 bullets) and then add a 1–2 sentence rationale. "
        "Do NOT invent citations; cite using the file name and chunk index shown in CONTEXT."
    )
    prompt = (
        f"SYSTEM:\n{system}\n\n"
        f"USER QUERY:\n{user_query}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "Write the answer now. If context is weak or off-topic, say so explicitly."
    )

    if HF_TOKEN:
        try:
            # Use the raw InferenceClient (supports many text models). If your env/model
            # only supports 'conversational', fall back gracefully.
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
            # Prefer text-generation endpoint for broad compatibility
            # (some hosted models don't support chat_completion)
            result = client.text_generation(
                prompt,
                max_new_tokens=400,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            answer = _clean(result)
            if not answer:
                answer = _fallback_answer(user_query, retrieved)
            return (answer, citations)
        except Exception as e:
            # Graceful fallback if model/endpoint isn’t compatible
            warn = f"(LLM unavailable: {e}) "
            return (warn + _fallback_answer(user_query, retrieved), citations)

    # 4) No HF/OpenAI configured → fallback
    return (_fallback_answer(user_query, retrieved), citations)
