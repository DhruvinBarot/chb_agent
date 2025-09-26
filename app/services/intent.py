from app.schemas import IntentResult
import re

# Very simple heuristics; replace with small LLM or classifier later.
def classify_intent(message: str) -> IntentResult:
    text = message.lower()

    if any(k in text for k in ["summarize", "summary", "synthesize"]):
        return IntentResult(intent="summarize", confidence=0.75)
    if any(k in text for k in ["compare", "contrast", "vs.", "versus"]):
        return IntentResult(intent="compare", confidence=0.7)
    if any(k in text for k in ["extract", "pull out", "list all", "what are the"]):
        return IntentResult(intent="extract", confidence=0.65)
    if any(k in text for k in ["cite", "citation", "references", "page"]):
        return IntentResult(intent="cite", confidence=0.6)
    if any(k in text for k in ["critique", "limitations", "reviewer", "revise"]):
        return IntentResult(intent="critique", confidence=0.6)

    return IntentResult(intent="other", confidence=0.4)

def normalize_message(message: str, intent: str) -> str:
    # Strip boilerplate, standardize whitespace, expand common acronyms (cheap stub).
    msg = re.sub(r"\s+", " ", message).strip()

    replacements = {
        "cpes": "Cannabis Pain Expectancies Scale",
        "eds": "Everyday Discrimination Scale",
        "act": "Acceptance and Commitment Therapy",
        "mturk": "Amazon Mechanical Turk",
    }
    for k, v in replacements.items():
        msg = re.sub(rf"\b{k}\b", v, msg, flags=re.IGNORECASE)

    # Intent-specific preambles (helps Step 3 retrieval later)
    if intent == "summarize":
        return f"Task: summarize. Scope: peer-reviewed PDFs in corpus. Query: {msg}."
    if intent == "compare":
        return f"Task: compare across papers/studies. Query: {msg}."
    if intent == "extract":
        return f"Task: extract structured facts (N, population, outcomes, effect direction). Query: {msg}."
    if intent == "cite":
        return f"Task: provide page-level citations supporting user query. Query: {msg}."
    if intent == "critique":
        return f"Task: critique methods/limitations. Query: {msg}."
    return msg
