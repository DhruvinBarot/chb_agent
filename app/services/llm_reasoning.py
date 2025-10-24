from huggingface_hub import InferenceClient
import os

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

def generate_answer(query: str, retrieved: list):
    """
    Generate a reasoned answer using retrieved text.
    Append citation list for traceability.
    """
    if retrieved:
        context_text = "\n\n".join(
            f"Source: {r['source']} (chunk {r['chunk']})\nExcerpt: {r['excerpt']}"
            for r in retrieved
        )
        citation_list = [f"{r['source']} (chunk {r['chunk']})" for r in retrieved]
    else:
        context_text = "General domain knowledge on pain, substance use, and behavioral health."
        citation_list = []

    prompt = (
        f"You are a research assistant summarizing evidence-based findings.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Provide a concise, evidence-grounded answer and end with a section titled 'Citations:' "
        f"listing the relevant paper names and chunk numbers."
    )

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        model_text = response.choices[0].message["content"].strip()
        return model_text, citation_list
    except Exception as e:
        return f"⚠️ LLM generation failed: {e}", citation_list
