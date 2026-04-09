"""
Custom NeMo Guardrails actions.

These are registered with the guardrails runtime and invoked by Colang flows.
Each action wraps a step in the RAG pipeline with appropriate validation.
"""

import json
import re
import logging
from typing import Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult

logger = logging.getLogger(__name__)


# ─── EXECUTION RAIL: RAG Search Action ──────────────────────────
@action(name="rag_search")
async def rag_search(query: str, context: Optional[dict] = None) -> ActionResult:
    """
    Perform hybrid search against Qdrant, then pass results through the LLM
    to generate a natural conversational response.
    """
    from rag_core.hybrid_qdrant_operations import HybridQdrantOperations
    from litellm import acompletion
    import os

    qdrant_ops = HybridQdrantOperations()

    try:
        results = qdrant_ops.hybrid_search(text=query, top_k=5)

        if not results:
            return ActionResult(
                return_value="I couldn't find any relevant movies in our database for that query. Could you try rephrasing?",
                context_updates={"retrieved_context": "No results found."},
            )

        # Format results as context for the LLM
        formatted_context = _format_movie_results(results)

        # Let the LLM generate a natural response based on the context
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        llm_response = await acompletion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly movie assistant. Based on the CONTEXT below, "
                        "answer the user's question in a natural, conversational tone. "
                        "Mention specific movie titles, ratings, directors, and cast when relevant. "
                        "Keep your response concise but informative. "
                        "Only use information from the CONTEXT — do not make up facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{formatted_context}\n\nUSER QUESTION: {query}",
                },
            ],
        )

        natural_answer = llm_response.choices[0].message.content

        return ActionResult(
            return_value=natural_answer,
            context_updates={"retrieved_context": formatted_context},
        )

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return ActionResult(
            return_value="I encountered an issue searching the database. Please try again.",
            context_updates={"retrieved_context": "Search error occurred."},
        )


# ─── INPUT RAIL: PII Masking Action ────────────────────────────
@action(name="mask_pii_action")
async def mask_pii_action(text: str) -> str:
    """
    Regex-based PII masking as a fast first pass before the LLM-based check.
    Covers emails, phone numbers, SSNs, and credit card numbers.
    """
    patterns = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    masked = text
    for pii_type, pattern in patterns.items():
        masked = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", masked)

    return masked


# ─── RETRIEVAL RAIL: Relevance Scoring ──────────────────────────
@action(name="check_chunk_relevance")
async def check_chunk_relevance(
    query: str, chunk: str, threshold: float = 0.4
) -> dict:
    """
    Quick heuristic relevance check for retrieved chunks.
    Returns whether the chunk passes the relevance threshold.
    """
    query_terms = set(query.lower().split())
    chunk_terms = set(chunk.lower().split())

    if not query_terms:
        return {"relevant": False, "score": 0.0}

    overlap = query_terms.intersection(chunk_terms)
    score = len(overlap) / len(query_terms)

    return {"relevant": score >= threshold, "score": round(score, 3)}


# ─── RETRIEVAL RAIL: Sensitive Document Filter ──────────────────
@action(name="filter_sensitive_chunks")
async def filter_sensitive_chunks(chunks: list) -> list:
    """
    Remove chunks that contain sensitive patterns (credentials, internal paths, etc.).
    """
    sensitive_patterns = [
        r"(api[_-]?key|secret|password|token)\s*[:=]",
        r"(mongodb|postgres|mysql|redis)://",
        r"/etc/(passwd|shadow|hosts)",
        r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
    ]

    compiled = [re.compile(p, re.IGNORECASE) for p in sensitive_patterns]

    safe_chunks = []
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else str(chunk)
        if not any(pat.search(text) for pat in compiled):
            safe_chunks.append(chunk)
        else:
            logger.warning("Filtered out a chunk containing sensitive patterns.")

    return safe_chunks


# ─── OUTPUT RAIL: Sensitive Data Removal ────────────────────────
@action(name="scrub_output")
async def scrub_output(response: str) -> str:
    """
    Final pass to remove any sensitive data that might have leaked into the output.
    """
    scrub_patterns = {
        "api_key": r"(sk-|pk_|rk_)[a-zA-Z0-9]{20,}",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "file_path": r"(/home/|/var/|/etc/|C:\\\\)[^\s]+",
    }

    scrubbed = response
    for data_type, pattern in scrub_patterns.items():
        scrubbed = re.sub(pattern, f"[REDACTED]", scrubbed)

    return scrubbed


# ─── HELPER: Format Movie Results ──────────────────────────────
def _format_movie_results(results: list) -> str:
    """Format Qdrant search results into a readable context string."""
    formatted_parts = []

    for i, item in enumerate(results, 1):
        parts = [f"Movie {i}:"]

        if "title" in item:
            parts.append(f"  Title: {item['title']}")
        if "genres" in item:
            genres = item["genres"]
            if isinstance(genres, str):
                try:
                    genres = json.loads(genres)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(genres, list):
                genre_names = [g["name"] if isinstance(g, dict) else str(g) for g in genres]
                parts.append(f"  Genres: {', '.join(genre_names)}")
            else:
                parts.append(f"  Genres: {genres}")
        if "vote_average" in item:
            parts.append(f"  Rating: {item['vote_average']}/10")
        if "release_date" in item:
            parts.append(f"  Released: {item['release_date']}")
        if "budget" in item and item["budget"]:
            parts.append(f"  Budget: ${item['budget']:,.0f}")
        if "revenue" in item and item["revenue"]:
            parts.append(f"  Revenue: ${item['revenue']:,.0f}")
        if "overview" in item:
            parts.append(f"  Overview: {item['overview']}")
        if "cast" in item:
            parts.append(f"  Cast: {item['cast']}")
        if "director" in item:
            parts.append(f"  Director: {item['director']}")

        formatted_parts.append("\n".join(parts))

    return "\n\n".join(formatted_parts)