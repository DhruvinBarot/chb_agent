# from datetime import datetime
# from fastapi import APIRouter, HTTPException

# from app.schemas import ChatRequest, ChatNormalized
# from app.services.intent import classify_intent, normalize_message
# from app.services.safety import basic_safety_check
# from app.utils.rate_limit import allow_request

# from app.memory.short_term import ShortTermMemory
# from app.memory.long_term import store_interaction, summarize_history

# from app.services.topics import is_domain_relevant, select_topic_terms
# from app.services.retrieval import retrieve_relevant_chunks, passes_relevance
# from app.services.llm_reasoning import generate_answer


# router = APIRouter()
# memory = ShortTermMemory(window_size=5)


# @router.post("/chat", response_model=ChatNormalized)
# def chat(req: ChatRequest):
#     # --- Rate limiting ---
#     if not allow_request(req.thread_id):
#         raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again shortly.")

#     # --- Safety check first ---
#     safety = basic_safety_check(req.message)
#     if not safety.allowed:
#         # remember the blocked interaction
#         memory.add(req.thread_id, "user", req.message)
#         memory.add(req.thread_id, "assistant", safety.replacement or "Blocked for safety.")

#         retrieved, _ = retrieve_relevant_chunks(req.message, set(), n_results=3)

#         context = memory.get(req.thread_id)
#         return ChatNormalized(
#             thread_id=req.thread_id,
#             message=req.message,
#             intent={"intent": "other", "confidence": 0.0},
#             safety=safety.model_dump(),
#             normalized_message=safety.replacement or "Blocked for safety.",
#             tags=["safety_blocked"],
#             context=context,
#             retrieval=[],
#             generated_answer=safety.replacement or "Blocked for safety.",
#             citations=[]
#         )

#     # --- Intent + normalization ---
#     intent = classify_intent(req.message)
#     normalized = normalize_message(req.message, intent.intent)

#     # --- Domain gate + topic terms (topic-agnostic) ---
#     domain_ok = is_domain_relevant(normalized)
#     topic_terms = select_topic_terms(normalized)

#     # --- Retrieval (topic-aware) ---
#     retrieved, max_score = retrieve_relevant_chunks(normalized, topic_terms, n_results=5)
#     relevant = passes_relevance(max_score)

#     # --- LLM reasoning with guardrails (refuse if OOD/low-evidence) ---
#     answer_text, citations = generate_answer(
#         user_query=normalized,
#         retrieved=retrieved,
#         domain_ok=domain_ok,
#         relevant=relevant
#     )

#     # --- Memory (short + long term) ---
#     memory.add(req.thread_id, "user", req.message)
#     memory.add(req.thread_id, "assistant", answer_text)
#     store_interaction(req.thread_id, req.message, answer_text, retrieved)
#     context = summarize_history(req.thread_id)  # or memory.get(req.thread_id)

#     # --- Logging (optional) ---
#     print("[USER_INPUT_EVENT]", {
#         "ts": datetime.utcnow().isoformat() + "Z",
#         "thread_id": req.thread_id,
#         "intent": intent.model_dump(),
#         "safety": safety.model_dump(),
#         "normalized_message": normalized,
#         "domain_ok": domain_ok,
#         "max_score": max_score
#     })

#     # --- Tags to help UI ---
#     tags = ["reasoned_response"]
#     if not domain_ok:
#         tags.append("out_of_domain")
#     if not relevant:
#         tags.append("low_evidence")

#     # --- Response ---
#     return ChatNormalized(
#         thread_id=req.thread_id,
#         message=req.message,
#         intent=intent.model_dump(),       # dict for Pydantic v2 validation
#         safety=safety.model_dump(),       # dict for Pydantic v2 validation
#         normalized_message=normalized,
#         tags=tags,
#         context=context,
#         retrieval=retrieved,
#         generated_answer=answer_text,
#         citations=citations
#     )


# app/services/retrieval.py


# Multi-Query Settings
MULTI_QUERY_ENABLED = True  # Toggle multi-query on/off
NUM_QUERY_VARIATIONS = 3    # How many variations to generate
USE_LLM_FOR_QUERIES = False # Use LLM (requires API key) vs templates

# Retrieval Settings
MAX_RESULTS = 12            # Results per query
MIN_SCORE = 0.25           # Minimum relevance threshold
TOP_N = 5     
import chromadb
from typing import List, Dict, Tuple, Optional
import re
import os
from collections import defaultdict

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_DIR = "data/chroma_db"
COLLECTION = "papers"

MAX_RESULTS = 12
MIN_SCORE = 0.25
TOP_N = 5
MULTI_QUERY_ENABLED = True
NUM_QUERY_VARIATIONS = 3

_whitespace = re.compile(r"\s+")

def _clean_excerpt(s: str) -> str:
    s = _whitespace.sub(" ", (s or "")).strip()
    return s.replace(" ,", ",").replace(" .", ".")

def _get_embedding_function():
    return SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def passes_relevance(max_score: float, threshold: float = MIN_SCORE) -> bool:
    """Check if the max score passes the relevance threshold."""
    return max_score >= threshold


# ============================================================
# STEP 1: MULTI-QUERY GENERATOR
# ============================================================

class MultiQueryGenerator:
    """
    Generates multiple query variations to improve retrieval recall.
    Based on the LangChain Multi-Query Retrieval pattern.
    """
    
    def __init__(self, num_variations: int = NUM_QUERY_VARIATIONS):
        self.num_variations = num_variations
        self.prompt_template = """Generate {n} different versions of this question to help retrieve relevant documents from a research paper database.

Each variation should rephrase the question in a different way while maintaining the core intent.

Original question: {question}

Output only the {n} alternative questions, one per line, without numbering or explanations."""

    def generate_with_llm(self, question: str) -> List[str]:
        """
        Generate query variations using an LLM (OpenAI/Anthropic).
        Falls back to template-based if LLM fails.
        """
        try:
            import openai
            
            prompt = self.prompt_template.format(
                n=self.num_variations,
                question=question
            )
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            # Parse response
            variations = response.choices[0].message.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip()]
            
            # Always include original question first
            all_queries = [question] + variations
            
            return all_queries[:self.num_variations + 1]
        
        except Exception as e:
            print(f"[MultiQueryGenerator] LLM generation failed: {e}")
            return self.generate_template_based(question)
    
    def generate_template_based(self, question: str) -> List[str]:
        """
        Fallback: Generate variations using templates.
        Fast and doesn't require API calls.
        """
        # Start with original
        variations = [question]
        
        # Template-based variations
        templates = [
            f"What are the key findings about {question}?",
            f"Explain the research on {question}",
            f"What information exists regarding {question}?",
            f"Summarize knowledge about {question}",
            f"What do papers say about {question}?"
        ]
        
        for template in templates:
            if len(variations) >= self.num_variations + 1:
                break
            # Avoid duplicates
            if template.lower().strip() != question.lower().strip():
                variations.append(template)
        
        return variations[:self.num_variations + 1]
    
    def generate(self, question: str, use_llm: bool = False) -> List[str]:
        """
        Main entry point for query generation.
        
        Args:
            question: Original user question
            use_llm: If True, use LLM; otherwise use templates
            
        Returns:
            List of query variations (including original)
        """
        if use_llm:
            return self.generate_with_llm(question)
        else:
            return self.generate_template_based(question)


# ============================================================
# STEP 2: MULTI-QUERY SEARCHER
# ============================================================

class MultiQuerySearcher:
    """
    Performs vector search with multiple query variations and merges results.
    Uses Reciprocal Rank Fusion (RRF) for result fusion.
    """
    
    def __init__(self, collection_name: str = COLLECTION, db_path: str = DB_DIR):
        self.collection_name = collection_name
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = _get_embedding_function()
    
    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        try:
            return self.client.get_collection(
                self.collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"[MultiQuerySearcher] Failed to get collection: {e}")
            return None
    
    def _create_doc_id(self, doc: Dict) -> str:
        """Create unique identifier for deduplication."""
        source = doc.get("source", "unknown")
        chunk = doc.get("chunk", -1)
        return f"{source}::{chunk}"
    
    def search_single(self, query: str, top_k: int = MAX_RESULTS) -> List[Dict]:
        """
        Search with a single query.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of document dictionaries with metadata
        """
        collection = self._get_collection()
        if collection is None:
            return []
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["distances", "metadatas", "documents"]
            )
            
            docs = (results.get("documents") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
            dists = (results.get("distances") or [[1.0] * len(docs)])[0]
            
            items = []
            for doc, meta, dist in zip(docs, metas, dists):
                score = max(0.0, 1.0 - float(dist))
                items.append({
                    "source": meta.get("source", "unknown.pdf"),
                    "chunk": meta.get("chunk", -1),
                    "excerpt": _clean_excerpt(doc)[:1400],
                    "score": round(score, 3),
                    "distance": round(float(dist), 3)
                })
            
            return items
        
        except Exception as e:
            print(f"[MultiQuerySearcher] Search failed for query '{query}': {e}")
            return []
    
    def search_multi(self, queries: List[str], top_k_per_query: int = MAX_RESULTS) -> List[Dict]:
        """
        Search with multiple queries and merge results using RRF.
        
        Args:
            queries: List of query variations
            top_k_per_query: How many results to get per query
            
        Returns:
            Merged and deduplicated list of documents
        """
        # Store results with their query ranks
        doc_results = defaultdict(list)  # doc_id -> list of (rank, score, doc)
        
        for query_idx, query in enumerate(queries):
            results = self.search_single(query, top_k=top_k_per_query)
            
            for rank, doc in enumerate(results):
                doc_id = self._create_doc_id(doc)
                doc_results[doc_id].append({
                    "rank": rank,
                    "score": doc.get("score", 0),
                    "doc": doc,
                    "query_idx": query_idx
                })
        
        # Apply Reciprocal Rank Fusion (RRF)
        merged = []
        for doc_id, occurrences in doc_results.items():
            # RRF formula: sum of 1/(k + rank) for each occurrence
            # k=60 is a common constant in RRF literature
            k = 60
            rrf_score = sum(1.0 / (k + occ["rank"] + 1) for occ in occurrences)
            
            # Get the best raw score
            max_score = max(occ["score"] for occ in occurrences)
            
            # Take the first occurrence's document and enhance it
            doc = occurrences[0]["doc"].copy()
            doc["score"] = round(max_score, 3)
            doc["rrf_score"] = round(rrf_score, 4)
            doc["query_hits"] = len(occurrences)  # How many queries found this
            
            merged.append(doc)
        
        # Sort by RRF score (better fusion than raw similarity)
        merged.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        
        return merged


# ============================================================
# STEP 3: COMPLETE MULTI-QUERY RAG RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(
    query: str, 
    topic_terms=None, 
    n_results: int = TOP_N,
    use_multi_query: bool = MULTI_QUERY_ENABLED,
    use_llm_for_queries: bool = False
) -> Tuple[List[Dict], float]:
    """
    Main retrieval function with multi-query support.
    
    Args:
        query: User's search query
        topic_terms: Optional topic terms (for future enhancement)
        n_results: Number of final results to return
        use_multi_query: Enable multi-query retrieval
        use_llm_for_queries: Use LLM for query generation (requires OpenAI API key)
        
    Returns:
        Tuple of (filtered_results, max_score)
    """
    
    if not use_multi_query:
        # Single query path (original behavior)
        items, max_score = retrieve(query)
        filtered = [item for item in items if item.get("score", 0) >= MIN_SCORE]
        return filtered[:n_results], max_score
    
    # ========== MULTI-QUERY PATH ==========
    
    print(f"[RETRIEVAL] 🔍 Multi-query retrieval for: '{query}'")
    
    # Step 1: Generate query variations
    generator = MultiQueryGenerator(num_variations=NUM_QUERY_VARIATIONS)
    queries = generator.generate(query, use_llm=use_llm_for_queries)
    
    print(f"[RETRIEVAL] 📝 Generated {len(queries)} query variations:")
    for i, q in enumerate(queries):
        print(f"  {i+1}. {q}")
    
    # Step 2: Search with all queries
    searcher = MultiQuerySearcher()
    merged_results = searcher.search_multi(queries, top_k_per_query=MAX_RESULTS)
    
    print(f"[RETRIEVAL] 📊 Found {len(merged_results)} unique documents")
    
    # Step 3: Filter by minimum score
    filtered = [
        item for item in merged_results 
        if item.get("score", 0) >= MIN_SCORE
    ]
    
    print(f"[RETRIEVAL] ✅ {len(filtered)} documents pass threshold (>= {MIN_SCORE})")
    
    # Step 4: Calculate max score
    max_score = max([item.get("score", 0) for item in merged_results], default=0.0)
    
    # Step 5: Return top N
    final_results = filtered[:n_results]
    
    print(f"[RETRIEVAL] 🎯 Returning top {len(final_results)} results")
    
    return final_results, max_score


def retrieve(query: str) -> Tuple[List[Dict], float]:
    """
    Legacy single-query retrieval (kept for backwards compatibility).
    """
    searcher = MultiQuerySearcher()
    items = searcher.search_single(query, top_k=MAX_RESULTS)
    
    if not items:
        return [], 0.0
    
    items.sort(key=lambda x: x["score"], reverse=True)
    max_score = max([item.get("score", 0) for item in items], default=0.0)
    
    return items, max_score