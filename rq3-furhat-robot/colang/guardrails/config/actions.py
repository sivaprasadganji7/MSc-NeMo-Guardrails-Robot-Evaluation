"""
guardrails/config/actions.py
─────────────────────────────
Custom actions auto-discovered by NeMo Guardrails.
"""

import os
import logging
import sys
from pathlib import Path

# Allow importing movie_kb_tmdb from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import movie_kb_tmdb as movie_kb

from nemoguardrails.actions import action

logger    = logging.getLogger("movie_guardrails_bot.actions")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))


@action(name="rag_search")
async def rag_search(query: str, context: dict = None) -> str:
    """
    Retrieve relevant movie docs from the KB and return a response prompt.
    NeMo will pass this string back into the LLM as the bot turn.
    """
    try:
        results = movie_kb.query_movies(query, n_results=RAG_TOP_K)
        if results and "documents" in results and results["documents"]:
            docs = [d for d in results["documents"][0] if d]
            if docs:
                logger.info("📚  Retrieved %d KB doc(s) for: %r", len(docs), query[:60])
                kb_context = "\n---\n".join(docs)
                return (
                    f"Using ONLY the following movie information to answer, "
                    f"respond in a warm, friendly tone:\n\n"
                    f"{kb_context}\n\n"
                    f"Question: {query}"
                )
    except Exception as exc:
        logger.error("rag_search error: %s", exc)

    logger.info("❌  No KB docs found for: %r", query[:60])
    return (
        "No specific movie information was found in the database for that query. "
        "Politely tell the user you don't have that information and suggest they "
        "try asking about a different movie or actor."
    )