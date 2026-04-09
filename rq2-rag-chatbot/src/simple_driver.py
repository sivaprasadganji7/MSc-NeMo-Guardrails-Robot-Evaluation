"""
Simple Movie Assistant — RAG + LLM with Multi-Turn Conversation (No Guardrails)
Now with experiment logging for human evaluation.

Uses:
  - Qdrant hybrid search (dense + sparse) for retrieval
  - LiteLLM + GPT-4o-mini for natural language generation
  - Redis for semantic caching
  - Conversation history for multi-turn context
  - JSON / TXT / CSV logging for experiment sessions
"""

import os
import csv
import json
import time
import asyncio
import datetime
from dotenv import load_dotenv, find_dotenv
from litellm import acompletion
import litellm
from litellm.caching import Cache

from rag_core.hybrid_qdrant_operations import HybridQdrantOperations

load_dotenv(find_dotenv())

SYSTEM_PROMPT = (
    "You are a friendly movie assistant powered by the TMDB 5000 dataset. "
    "Answer the user's question based on the CONTEXT provided and the conversation history. "
    "Mention specific movie titles, ratings, directors, and cast when relevant. "
    "Be concise but informative. Only use information from the CONTEXT. "
    "If the user asks a follow-up about a movie already discussed, use the conversation history to maintain continuity. "
    "If the CONTEXT doesn't have relevant results for the query, say so honestly."
)

LOG_DIR = "experiment_logs"


class ExperimentLogger:
    def __init__(self, participant_id: str, condition: str = "baseline_no_guardrails", log_dir: str = LOG_DIR):
        self.participant_id = participant_id
        self.condition = condition
        self.log_dir = log_dir
        self.session_start = datetime.datetime.now()
        self.turns = []
        self.turn_count = 0

        os.makedirs(self.log_dir, exist_ok=True)

        timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
        safe_pid = self.participant_id.replace(" ", "_")

        self.json_file = os.path.join(
            self.log_dir, f"{safe_pid}_{self.condition}_{timestamp_str}.json"
        )
        self.txt_file = os.path.join(
            self.log_dir, f"{safe_pid}_{self.condition}_{timestamp_str}.txt"
        )

        with open(self.txt_file, "w", encoding="utf-8") as f:
            f.write(f"Experiment Log - Participant {self.participant_id}\n")
            f.write(f"Condition: {self.condition}\n")
            f.write(f"Started: {self.session_start.isoformat()}\n")
            f.write("=" * 70 + "\n\n")

    def log_turn(
        self,
        user_input: str,
        bot_response: str,
        response_time_ms: int | None = None,
        kb_docs_retrieved: int = 0,
        retrieved_titles: list[str] | None = None,
        cache_status: str | None = None,
        history_size: int | None = None,
        raw_context: str | None = None,
    ):
        self.turn_count += 1
        now = datetime.datetime.now()
        elapsed = (now - self.session_start).total_seconds()

        turn_data = {
            "turn_number": self.turn_count,
            "timestamp": now.isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "user_input": user_input,
            "bot_response": bot_response,
            "response_time_ms": response_time_ms,
            "kb_docs_retrieved": kb_docs_retrieved,
            "retrieved_titles": retrieved_titles or [],
            "cache_status": cache_status,
            "history_size_messages": history_size,
            "raw_context": raw_context,
        }
        self.turns.append(turn_data)

        with open(self.txt_file, "a", encoding="utf-8") as f:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            f.write(f"[Turn {self.turn_count} | {mins}m {secs}s | {response_time_ms}ms]\n")
            f.write(f"User: {user_input}\n")
            f.write(f"Bot:  {bot_response}\n")
            f.write(f"KB docs retrieved: {kb_docs_retrieved}\n")
            if retrieved_titles:
                f.write(f"Retrieved titles: {', '.join(retrieved_titles)}\n")
            if cache_status:
                f.write(f"Cache status: {cache_status}\n")
            if history_size is not None:
                f.write(f"History size: {history_size} messages\n")
            f.write("\n")

    def save(self, assistant_state: dict | None = None):
        session_end = datetime.datetime.now()
        duration = (session_end - self.session_start).total_seconds()

        valid_times = [t["response_time_ms"] for t in self.turns if t["response_time_ms"] is not None]
        avg_response_time_ms = round(sum(valid_times) / len(valid_times), 1) if valid_times else 0.0

        session_data = {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "session_start": self.session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "duration_seconds": round(duration, 1),
            "total_turns": self.turn_count,
            "avg_response_time_ms": avg_response_time_ms,
            "turns": self.turns,
        }

        if assistant_state:
            session_data["final_state"] = assistant_state

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        with open(self.txt_file, "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Session ended: {session_end.isoformat()}\n")
            f.write(f"Total duration: {round(duration, 1)}s\n")
            f.write(f"Total turns: {self.turn_count}\n")
            f.write(f"Average response time: {avg_response_time_ms}ms\n")
            if assistant_state:
                f.write("\nFinal state:\n")
                f.write(json.dumps(assistant_state, indent=2, ensure_ascii=False))
                f.write("\n")

        summary_csv = os.path.join(self.log_dir, "all_sessions_summary.csv")
        file_exists = os.path.exists(summary_csv)

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "participant_id",
                    "condition",
                    "start_time",
                    "duration_seconds",
                    "total_turns",
                    "avg_response_time_ms",
                ])
            writer.writerow([
                self.participant_id,
                self.condition,
                self.session_start.isoformat(),
                round(duration, 1),
                self.turn_count,
                avg_response_time_ms,
            ])

        print("\n✅ Logs saved:")
        print(f"   JSON: {self.json_file}")
        print(f"   TXT:  {self.txt_file}")
        print(f"   CSV:  {summary_csv}")


class SimpleMovieAssistant:
    def __init__(self):
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = os.environ.get("REDIS_PORT", "6379")
        litellm.cache = Cache(type="redis", host=redis_host, port=redis_port)

        self.qdrant = HybridQdrantOperations()
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        self.history: list[dict] = []
        self.max_history_turns = 10

        print("✅ Simple Movie Assistant initialized (multi-turn, no guardrails)")

    async def ask(self, question: str) -> tuple[str, dict]:
        start_time = time.perf_counter()

        # Step 1: Retrieve relevant movies from Qdrant
        results = self.qdrant.hybrid_search(text=question, top_k=5)

        # Step 2: Format context
        context = self._format_results(results) if results else "No results found."

        # Step 3: Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)

        user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        messages.append({"role": "user", "content": user_message})

        # Step 4: Send to LLM
        response = await acompletion(
            model=self.model,
            messages=messages,
            cache={"no-cache": False, "no-store": False},
        )

        answer = response.choices[0].message.content

        # Step 5: Save conversation history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-(self.max_history_turns * 2):]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000)

        titles = []
        if results:
            for item in results:
                title = item.get("title")
                if title:
                    titles.append(str(title))

        metadata = {
            "response_time_ms": elapsed_ms,
            "kb_docs_retrieved": len(results) if results else 0,
            "retrieved_titles": titles,
            "history_size_messages": len(self.history),
            "raw_context": context,
            "cache_status": self._extract_cache_status(response),
        }

        return answer, metadata

    def clear_history(self):
        self.history.clear()
        print("🗑️ Conversation history cleared.")

    def get_state(self) -> dict:
        return {
            "model": self.model,
            "history_size_messages": len(self.history),
            "max_history_turns": self.max_history_turns,
            "history": self.history,
        }

    def _extract_cache_status(self, response) -> str | None:
        """
        Tries to extract cache-related metadata if present.
        LiteLLM caching controls exist, but cache-hit fields may vary by provider/setup.
        """
        try:
            if hasattr(response, "_hidden_params"):
                hidden = getattr(response, "_hidden_params", {}) or {}
                for key in ["cache_hit", "cache_key", "cached_response"]:
                    if key in hidden:
                        return f"{key}={hidden[key]}"
            return None
        except Exception:
            return None

    def _format_results(self, results: list) -> str:
        parts = []
        for i, item in enumerate(results, 1):
            lines = [f"Movie {i}:"]
            for key in [
                "title", "overview", "genres", "cast", "director",
                "release_date", "vote_average", "budget", "revenue"
            ]:
                if key in item and item[key]:
                    val = item[key]
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    lines.append(f"  {key}: {val}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


async def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   🎬 Simple Movie Assistant (Multi-Turn, No Guardrails)     ║
║   RAG + LLM — Qdrant + GPT-4o-mini + History + Logging      ║
╠══════════════════════════════════════════════════════════════╣
║ Commands:                                                   ║
║   'bye'   — exit                                            ║
║   'clear' — reset conversation history                      ║
╚══════════════════════════════════════════════════════════════╝
""")

    participant_id = input("Enter participant ID (e.g., P01): ").strip() or "P00"
    condition = input("Enter condition name [baseline_no_guardrails]: ").strip() or "baseline_no_guardrails"

    assistant = SimpleMovieAssistant()
    logger = ExperimentLogger(participant_id=participant_id, condition=condition)

    while True:
        try:
            question = input("🎬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("bye", "exit", "quit"):
            print("👋 Goodbye!")
            break

        if question.lower() == "clear":
            assistant.clear_history()
            continue

        print("\n⏳ Searching...\n")

        try:
            answer, meta = await assistant.ask(question)

            logger.log_turn(
                user_input=question,
                bot_response=answer,
                response_time_ms=meta.get("response_time_ms"),
                kb_docs_retrieved=meta.get("kb_docs_retrieved", 0),
                retrieved_titles=meta.get("retrieved_titles"),
                cache_status=meta.get("cache_status"),
                history_size=meta.get("history_size_messages"),
                raw_context=meta.get("raw_context"),
            )

            print(f"🤖 Assistant: {answer}\n")

        except Exception as e:
            error_reply = "Sorry, I ran into a problem while searching for movie information."
            logger.log_turn(
                user_input=question,
                bot_response=error_reply,
                response_time_ms=None,
                kb_docs_retrieved=0,
                retrieved_titles=[],
                cache_status=None,
                history_size=len(assistant.history),
                raw_context=None,
            )
            print(f"❗ Error: {e}")
            print(f"🤖 Assistant: {error_reply}\n")

    logger.save(assistant_state=assistant.get_state())


if __name__ == "__main__":
    asyncio.run(main())