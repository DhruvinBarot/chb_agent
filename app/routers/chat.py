from fastapi import APIRouter, HTTPException
from app.schemas import ChatRequest, ChatNormalized
from app.services.intent import classify_intent, normalize_message
from app.services.safety import basic_safety_check
from app.utils.rate_limit import allow_request
from datetime import datetime
from app.memory.short_term import ShortTermMemory
from app.services.retrieval import retrieve_relevant_chunks
from app.services.llm_reasoning import generate_answer  # step 4 import
from app.memory.long_term import store_interaction, summarize_history # step 5 import


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
        memory.add(req.thread_id, "user", req.message)
        memory.add(req.thread_id, "assistant", safety.replacement or "Blocked for safety")

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

    # Intent classification + normalization
    intent = classify_intent(req.message)
    normalized = normalize_message(req.message, intent.intent)

    # Retrieve relevant paper chunks
    retrieved = retrieve_relevant_chunks(req.message, n_results=3)

    # 🧠 Step 4: LLM reasoning with retrieved context
    # ✅ Unpack the tuple properly
    answer_text, citations = generate_answer(normalized, retrieved, req.thread_id)
    
    memory.add(req.thread_id, "user", req.message)
    memory.add(req.thread_id, "assistant", answer_text)

    # ✅ Store interaction in long-term memory
    store_interaction(req.thread_id, req.message, answer_text, retrieved)

    context = summarize_history(req.thread_id)

    # Logging (optional)
    event = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "thread_id": req.thread_id,
        "intent": intent.model_dump(),
        "safety": safety.model_dump(),
        "normalized_message": normalized,
        "context_window": context,
        "retrieval": retrieved,
        "generated_answer": answer_text,
        "citations": citations,
    }
    print("[USER_INPUT_EVENT]", event)

    # Final response
    return ChatNormalized(
        thread_id=req.thread_id,
        message=req.message,
        intent=intent.model_dump(),     # ✅ dict
        safety=safety.model_dump(), 
        normalized_message=normalized,
        tags=["reasoned_response"],
        context=context,
        retrieval=retrieved,
        generated_answer=answer_text,  # ✅ string only
        citations=citations              # ✅ separate list
    )
