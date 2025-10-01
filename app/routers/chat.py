from fastapi import APIRouter, HTTPException
from app.schemas import ChatRequest, ChatNormalized
from app.services.intent import classify_intent, normalize_message
from app.services.safety import basic_safety_check
from app.utils.rate_limit import allow_request
from datetime import datetime
from app.memory.short_term import ShortTermMemory
from app.services.retrieval import retrieve_relevant_chunks

router = APIRouter()
memory = ShortTermMemory(window_size=5)


@router.post("/chat", response_model=ChatNormalized)
def chat(req: ChatRequest):
    # Basic in-memory rate-limit
    if not allow_request(req.thread_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again shortly.")

    # Safety first
    safety = basic_safety_check(req.message)

    if not safety.allowed:
        # Store blocked input + response in memory
        memory.add(req.thread_id, "user", req.message)
        memory.add(req.thread_id, "assistant", safety.replacement or "Blocked for safety")

        # Even for blocked messages, attempt retrieval (optional)
        retrieved = retrieve_relevant_chunks(req.message, n_results=3)

        return ChatNormalized(
            thread_id=req.thread_id,
            message=req.message,
            intent={"intent": "other", "confidence": 0.0},
            safety=safety,
            normalized_message=safety.replacement
                or "Your message cannot be processed as-is due to safety constraints.",
            tags=["safety_blocked"],
            context=memory.get(req.thread_id),
            retrieval=retrieved
        )

    # Intent routing + normalization
    intent = classify_intent(req.message)
    normalized = normalize_message(req.message, intent.intent)

    # Retrieval
    retrieved = retrieve_relevant_chunks(req.message, n_results=3)

    # Add input + placeholder response to memory
    memory.add(req.thread_id, "user", req.message)
    assistant_msg = f"Noted intent: {intent.intent}"
    memory.add(req.thread_id, "assistant", assistant_msg)

    context = memory.get(req.thread_id)

    # Example logging payload (replace with real logger / Langfuse later)
    event = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "thread_id": req.thread_id,
        "intent": intent.model_dump(),
        "safety": safety.model_dump(),
        "normalized_message": normalized,
        "context_window": context,
        "retrieval": retrieved
    }
    print("[USER_INPUT_EVENT]", event)

    return ChatNormalized(
        thread_id=req.thread_id,
        message=req.message,
        intent=intent,
        safety=safety,
        normalized_message=normalized,
        tags=["validated", "ready_for_retrieval"],
        context=context,
        retrieval=retrieved
    )
