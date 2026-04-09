"""
RAG Operations with NeMo Guardrails.

Orchestrates the full pipeline:
  User Query → Input Rails → Cache Check → Retrieval + Retrieval Rails
  → Dialog Rails → LLM Generation → Execution Rails → Output Rails → Response

All rail stages are handled by NeMo Guardrails config (no hardcoding).
"""

import os
import json
import logging
from typing import Union

import litellm
from litellm import completion
from litellm.caching import Cache
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper
from nemoguardrails import RailsConfig, LLMRails
from dotenv import load_dotenv, find_dotenv

from rag_core.hybrid_qdrant_operations import HybridQdrantOperations
from utils.decorators import compute_execution_time

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


class RAGOperations:
    """
    Full RAG pipeline with NeMo Guardrails at every stage.

    Rail stages (all configured in config/, not hardcoded):
    ┌─────────────────┬─────────────────────────────────────────────┐
    │ Stage           │ What it does                                │
    ├─────────────────┼─────────────────────────────────────────────┤
    │ Input Rails     │ Content safety, jailbreak, topic, PII      │
    │ Retrieval Rails │ Chunk relevance, sensitive doc filtering    │
    │ Dialog Rails    │ Colang flow control, intent mapping         │
    │ Execution Rails │ Action I/O validation                       │
    │ Output Rails    │ Safety check, hallucination, data scrubbing │
    └─────────────────┴─────────────────────────────────────────────┘
    """

    ResponseType = Union[ModelResponse, CustomStreamWrapper]

    def __init__(self):
        # ── Redis semantic cache ────────────────────────────────
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = os.environ.get("REDIS_PORT", "6379")
        litellm.cache = Cache(type="redis", host=redis_host, port=redis_port)

        # ── NeMo Guardrails ─────────────────────────────────────
        config_path = os.path.join(os.path.dirname(__file__), "..", "config")
        self.rails_config = RailsConfig.from_path(config_path)
        self.rails = LLMRails(config=self.rails_config)

        # Register custom actions with the guardrails runtime
        self._register_actions()

        # ── Qdrant (used inside the rag_search action) ──────────
        self.qdrant_ops = HybridQdrantOperations()

        logger.info("RAGOperations initialized with NeMo Guardrails")

    def _register_actions(self):
        """Register custom Python actions from config/actions.py."""
        from config.actions import (
            rag_search,
            mask_pii_action,
            check_chunk_relevance,
            filter_sensitive_chunks,
            scrub_output,
        )

        self.rails.register_action(rag_search, name="rag_search")
        self.rails.register_action(mask_pii_action, name="mask_pii_action")
        self.rails.register_action(check_chunk_relevance, name="check_chunk_relevance")
        self.rails.register_action(filter_sensitive_chunks, name="filter_sensitive_chunks")
        self.rails.register_action(scrub_output, name="scrub_output")

    @compute_execution_time
    async def process_query(self, user_query: str) -> dict:
        """
        Main entry point. Sends the user query through the full
        NeMo Guardrails pipeline (all 5 rail stages).

        The guardrails runtime automatically applies:
          1. Input rails   (from config.yml → input_rails)
          2. Dialog rails  (from rails.co → Colang flows)
          3. Retrieval rails (during rag_search action)
          4. Execution rails (wrapping action calls)
          5. Output rails  (from config.yml → output_rails)
        """
        try:
            # NeMo Guardrails handles the entire pipeline
            response = await self.rails.generate_async(
                messages=[{"role": "user", "content": user_query}]
            )

            return {
                "status": "success",
                "response": response.get("content", response) if isinstance(response, dict) else str(response),
                "rail_log": self._get_rail_log(),
            }

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {
                "status": "error",
                "response": "I encountered an issue processing your request. Please try again.",
                "error": str(e),
            }

    @compute_execution_time
    def process_query_sync(self, user_query: str) -> dict:
        """Synchronous wrapper for process_query."""
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_query(user_query))
        finally:
            loop.close()

    @compute_execution_time
    def direct_rag_query(self, user_query: str) -> dict:
        """
        Bypass guardrails for testing/debugging.
        Performs retrieval + LLM completion directly.
        """
        # Retrieve context
        results = self.qdrant_ops.hybrid_search(text=user_query, top_k=5)
        context = self._format_context(results)

        # Build prompt
        prompt = self._build_prompt(user_query, context)

        # LLM completion with caching
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a movie expert assistant. Answer the USER_QUERY based on "
                    "the CONTEXT provided. If the context doesn't contain enough information, "
                    "say so honestly. Do not fabricate information."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self._chat_completion(messages)

        return {
            "status": "success",
            "response": response,
            "context_used": len(results),
        }

    def _build_prompt(self, user_query: str, context: str) -> str:
        """Build the RAG prompt with retrieved context."""
        return f"""
<CONTEXT>
{context}
</CONTEXT>

<USER_QUERY>
{user_query}
</USER_QUERY>
"""

    def _format_context(self, results: list) -> str:
        """Format search results into context string."""
        parts = []
        for i, item in enumerate(results, 1):
            entry = [f"[Movie {i}]"]
            for key in ["title", "overview", "genres", "cast", "director",
                        "release_date", "vote_average", "budget", "revenue"]:
                if key in item and item[key]:
                    val = item[key]
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    entry.append(f"  {key}: {val}")
            parts.append("\n".join(entry))
        return "\n\n".join(parts)

    @compute_execution_time
    def _chat_completion(self, messages: list) -> str:
        """LiteLLM completion with Redis caching."""
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        response = completion(
            model=model,
            messages=messages,
            cache={"no-cache": False, "no-store": False},
        )

        return response.choices[0].message.content

    def _get_rail_log(self) -> list:
        """Extract the guardrails execution log for debugging."""
        try:
            log = self.rails.explain()
            return log.colang_history if hasattr(log, "colang_history") else []
        except Exception:
            return []

    def get_health(self) -> dict:
        """Health check for the RAG system."""
        health = {"guardrails": "ok", "qdrant": "unknown", "cache": "unknown"}

        try:
            info = self.qdrant_ops.get_collection_info()
            health["qdrant"] = info
        except Exception as e:
            health["qdrant"] = f"error: {e}"

        try:
            import redis as redis_lib
            r = redis_lib.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
            )
            r.ping()
            health["cache"] = "ok"
        except Exception as e:
            health["cache"] = f"error: {e}"

        return health