from huggingface_hub import InferenceClient
import os

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

def generate_answer(query: str, retrieved: list):
    """
    Generate an LLM answer that reasons over retrieved text.
    If retrieval is empty, provide a general, knowledge-based response.
    """
    # ✅ Combine retrieved chunks into context
    if retrieved:
        context_text = "\n\n".join(
            f"[{r.get('source', 'unknown')}] {r.get('content', '')}" for r in retrieved
        )
        prompt = (
            f"You are a helpful research assistant. "
            f"Use the following context from academic papers to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely and clearly:"
        )
    else:
        # 🧩 fallback prompt if retrieval returns nothing
        prompt = (
            f"You are a research assistant specialized in health behavior. "
            f"Answer this based on general scientific understanding:\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely and clearly:"
        )

    # Generate a response using the HF model
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"⚠️ LLM generation failed: {e}"
