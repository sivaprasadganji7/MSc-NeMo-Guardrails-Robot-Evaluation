"""
Interactive driver for the Guardrailed RAG Application.
Supports multi-turn conversations with conversation history.
Now includes JSON / TXT / CSV experiment logging for human evaluation.

Usage:
    python driver.py              # Interactive mode
    python driver.py --health     # Health check
    python driver.py --direct     # Bypass guardrails (debug mode)
"""

import sys
import os
import csv
import json
import time
import asyncio
import logging
import datetime

from rag_core.rag_ops import RAGOperations
from rag_core.hybrid_qdrant_operations import HybridQdrantOperations
from litellm import acompletion

# ── Logging Setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("driver")

LOG_DIR = "experiment_logs"

# ── Conversation History ────────────────────────────────────────
conversation_history: list[dict] = []
MAX_HISTORY_TURNS = 10


class ExperimentLogger:
    def __init__(self, participant_id: str, condition: str = "guardrails", log_dir: str = LOG_DIR):
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
        guardrail_triggered: bool = False,
        guardrail_type: str | None = None,
        guardrail_status: str | None = None,
        kb_docs_retrieved: int = 0,
        retrieved_titles: list[str] | None = None,
        history_size: int | None = None,
        rails_stats: dict | None = None,
        direct_mode: bool = False,
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
            "guardrail_triggered": guardrail_triggered,
            "guardrail_type": guardrail_type,
            "guardrail_status": guardrail_status,
            "kb_docs_retrieved": kb_docs_retrieved,
            "retrieved_titles": retrieved_titles or [],
            "history_size_messages": history_size,
            "rails_stats": rails_stats,
            "direct_mode": direct_mode,
        }
        self.turns.append(turn_data)

        with open(self.txt_file, "a", encoding="utf-8") as f:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            f.write(f"[Turn {self.turn_count} | {mins}m {secs}s | {response_time_ms}ms]\n")
            f.write(f"User: {user_input}\n")
            f.write(f"Bot:  {bot_response}\n")
            f.write(f"Direct mode: {direct_mode}\n")
            f.write(f"Guardrail status: {guardrail_status}\n")
            f.write(f"Guardrail triggered: {guardrail_triggered}\n")
            if guardrail_type:
                f.write(f"Guardrail type: {guardrail_type}\n")
            f.write(f"KB docs retrieved: {kb_docs_retrieved}\n")
            if retrieved_titles:
                f.write(f"Retrieved titles: {', '.join(retrieved_titles)}\n")
            if history_size is not None:
                f.write(f"History size: {history_size} messages\n")
            if rails_stats:
                f.write(f"Rails stats: {json.dumps(rails_stats, ensure_ascii=False)}\n")
            f.write("\n")

    def save(self, final_state: dict | None = None):
        session_end = datetime.datetime.now()
        duration = (session_end - self.session_start).total_seconds()

        valid_times = [t["response_time_ms"] for t in self.turns if t["response_time_ms"] is not None]
        avg_response_time_ms = round(sum(valid_times) / len(valid_times), 1) if valid_times else 0.0
        guardrail_triggers = sum(1 for t in self.turns if t["guardrail_triggered"])

        session_data = {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "session_start": self.session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "duration_seconds": round(duration, 1),
            "total_turns": self.turn_count,
            "guardrail_triggers": guardrail_triggers,
            "avg_response_time_ms": avg_response_time_ms,
            "turns": self.turns,
        }

        if final_state:
            session_data["final_state"] = final_state

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        with open(self.txt_file, "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Session ended: {session_end.isoformat()}\n")
            f.write(f"Total duration: {round(duration, 1)}s\n")
            f.write(f"Total turns: {self.turn_count}\n")
            f.write(f"Guardrail triggers: {guardrail_triggers}\n")
            f.write(f"Average response time: {avg_response_time_ms}ms\n")
            if final_state:
                f.write("\nFinal state:\n")
                f.write(json.dumps(final_state, indent=2, ensure_ascii=False))
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
                    "guardrail_triggers",
                    "avg_response_time_ms",
                ])
            writer.writerow([
                self.participant_id,
                self.condition,
                self.session_start.isoformat(),
                round(duration, 1),
                self.turn_count,
                guardrail_triggers,
                avg_response_time_ms,
            ])

        print("\n✅ Logs saved:")
        print(f"   JSON: {self.json_file}")
        print(f"   TXT:  {self.txt_file}")
        print(f"   CSV:  {summary_csv}")


def print_banner():
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║     🎬  TMDB Movie RAG with NeMo Guardrails (Multi-Turn)    ║
╠══════════════════════════════════════════════════════════════╣
║  Rail Stages Active:                                        ║
║    ① Input Rails     → Safety, jailbreak, topic, PII        ║
║    ② Retrieval Rails → Chunk relevance, doc filtering       ║
║    ③ Dialog Rails    → Colang flow control                  ║
║    ④ Execution Rails → Action I/O validation                ║
║    ⑤ Output Rails    → Fact check, data scrub, safety       ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                  ║
║    'bye'/'exit' → quit  │  'health' → status                ║
║    'clear'      → reset conversation history                ║
╚══════════════════════════════════════════════════════════════╝
"""
    )


def detect_guardrail_type(response_text: str) -> str | None:
    if not response_text:
        return None

    text = response_text.lower()

    patterns = {
        "off_topic": [
            "movie assistant",
            "specifically designed",
            "helpful movie assistant",
        ],
        "jailbreak_or_safety": [
            "i'm sorry, i can't respond",
            "sorry, i cannot process",
            "cannot process",
            "can't respond",
        ],
        "generic_block": [
            "i appreciate your curiosity",
        ],
    }

    for label, phrases in patterns.items():
        if any(p in text for p in phrases):
            return label

    return None


def format_retrieved_context(results: list) -> tuple[str, list[str]]:
    context_parts = []
    titles = []

    for i, item in enumerate(results, 1):
        lines = [f"Movie {i}:"]
        title = item.get("title")
        if title:
            titles.append(str(title))

        for key in [
            "title", "overview", "genres", "cast", "director",
            "release_date", "vote_average", "budget", "revenue"
        ]:
            if key in item and item[key]:
                val = item[key]
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                lines.append(f"  {key}: {val}")
        context_parts.append("\n".join(lines))

    return "\n\n".join(context_parts), titles


async def generate_natural_response(user_query: str, retrieved_context: str) -> str:
    """
    Pass retrieved context + conversation history through the LLM
    to generate a natural, conversational response.
    """
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly movie assistant powered by the TMDB 5000 dataset. "
                "Answer the user's question based on the CONTEXT provided and the conversation history. "
                "Mention specific movie titles, ratings, directors, and cast when relevant. "
                "Be concise but informative. Only use information from the CONTEXT. "
                "If the user asks a follow-up about a movie already discussed, use the conversation history to maintain continuity. "
                "If the CONTEXT doesn't have relevant results for the query, say so honestly."
            ),
        },
    ]

    messages.extend(conversation_history)

    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{retrieved_context}\n\nUSER QUESTION: {user_query}",
    })

    response = await acompletion(model=model, messages=messages)
    return response.choices[0].message.content


async def process_with_guardrails_and_llm(rag_ops: RAGOperations, user_input: str) -> tuple[str, dict]:
    """
    Run through NeMo Guardrails pipeline, then generate a natural LLM response.
    Returns (response_text, metadata_dict).
    """
    global conversation_history

    metadata = {
        "guardrail_triggered": False,
        "guardrail_type": None,
        "guardrail_status": None,
        "kb_docs_retrieved": 0,
        "retrieved_titles": [],
        "rails_stats": None,
        "history_size_messages": len(conversation_history),
    }

    # Step 1: Run through guardrails pipeline
    result = await rag_ops.process_query(user_input)
    status = result.get("status", "unknown")
    response = result.get("response", "")

    metadata["guardrail_status"] = status

    # NeMo-style detailed stats may be available in guardrails_data.log.stats
    guardrails_data = result.get("guardrails_data", {})
    if isinstance(guardrails_data, dict):
        log_data = guardrails_data.get("log", {})
        if isinstance(log_data, dict):
            metadata["rails_stats"] = log_data.get("stats")

    if status != "success":
        metadata["guardrail_triggered"] = True
        metadata["guardrail_type"] = "pipeline_error_or_block"
        return response or "The request could not be processed safely.", metadata

    # Step 2: Detect blocking replies and do not add blocked turns to history
    blocking_type = detect_guardrail_type(response)
    if blocking_type is not None:
        metadata["guardrail_triggered"] = True
        metadata["guardrail_type"] = blocking_type
        return response, metadata

    # Step 3: If passed, retrieve documents and generate natural response
    qdrant_ops = HybridQdrantOperations()
    results = qdrant_ops.hybrid_search(text=user_input, top_k=5)

    metadata["kb_docs_retrieved"] = len(results) if results else 0

    if not results:
        natural_response = (
            "I couldn't find any relevant movies in our database for that query. "
            "Could you try rephrasing?"
        )
    else:
        context, titles = format_retrieved_context(results)
        metadata["retrieved_titles"] = titles
        natural_response = await generate_natural_response(user_input, context)

    # Step 4: Save only successful allowed turns into history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": natural_response})

    if len(conversation_history) > MAX_HISTORY_TURNS * 2:
        conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    metadata["history_size_messages"] = len(conversation_history)

    return natural_response, metadata


async def interactive_loop(rag_ops: RAGOperations, logger_obj: ExperimentLogger, direct_mode: bool = False):
    """Main interactive conversation loop with multi-turn support."""
    global conversation_history
    print_banner()

    while True:
        try:
            user_input = input("\n🎬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("bye", "exit", "quit"):
            print("👋 Goodbye! Hope you found something great to watch!")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            print("🗑️ Conversation history cleared.")
            continue

        if user_input.lower() == "health":
            health = rag_ops.get_health()
            print("\n📊 System Health:")
            for component, status in health.items():
                icon = "✅" if status == "ok" or (isinstance(status, dict) and "error" not in str(status)) else "❌"
                print(f"   {icon} {component}: {status}")
            continue

        print("\n⏳ Processing through guardrail pipeline...\n")
        start_time = time.perf_counter()

        try:
            if direct_mode:
                result = rag_ops.direct_rag_query(user_input)
                response = result.get("response", "No response generated.")
                meta = {
                    "guardrail_triggered": False,
                    "guardrail_type": None,
                    "guardrail_status": "direct_mode",
                    "kb_docs_retrieved": 0,
                    "retrieved_titles": [],
                    "history_size_messages": len(conversation_history),
                    "rails_stats": None,
                }
            else:
                response, meta = await process_with_guardrails_and_llm(rag_ops, user_input)

            response_time_ms = round((time.perf_counter() - start_time) * 1000)

            logger_obj.log_turn(
                user_input=user_input,
                bot_response=response,
                response_time_ms=response_time_ms,
                guardrail_triggered=meta.get("guardrail_triggered", False),
                guardrail_type=meta.get("guardrail_type"),
                guardrail_status=meta.get("guardrail_status"),
                kb_docs_retrieved=meta.get("kb_docs_retrieved", 0),
                retrieved_titles=meta.get("retrieved_titles", []),
                history_size=meta.get("history_size_messages"),
                rails_stats=meta.get("rails_stats"),
                direct_mode=direct_mode,
            )

            print(f"🤖 Assistant: {response}")

        except Exception as e:
            response_time_ms = round((time.perf_counter() - start_time) * 1000)
            error_reply = "Sorry, I ran into a problem while processing your request."

            logger_obj.log_turn(
                user_input=user_input,
                bot_response=error_reply,
                response_time_ms=response_time_ms,
                guardrail_triggered=True,
                guardrail_type="runtime_error",
                guardrail_status="error",
                kb_docs_retrieved=0,
                retrieved_titles=[],
                history_size=len(conversation_history),
                rails_stats=None,
                direct_mode=direct_mode,
            )

            logger.exception("Error during interactive processing")
            print(f"🤖 Assistant: {error_reply}")


def main():
    global conversation_history

    direct_mode = "--direct" in sys.argv

    print("🔧 Initializing RAG system...")
    rag_ops = RAGOperations()

    if "--health" in sys.argv:
        health = rag_ops.get_health()
        print("\n📊 System Health:")
        for component, status in health.items():
            print(f"   • {component}: {status}")
        return

    participant_id = input("Enter participant ID (e.g., P01): ").strip() or "P00"
    condition_default = "guardrails_direct" if direct_mode else "guardrails"
    condition = input(f"Enter condition name [{condition_default}]: ").strip() or condition_default

    logger_obj = ExperimentLogger(participant_id=participant_id, condition=condition)

    try:
        asyncio.run(interactive_loop(rag_ops, logger_obj=logger_obj, direct_mode=direct_mode))
    finally:
        final_state = {
            "history_size_messages": len(conversation_history),
            "max_history_turns": MAX_HISTORY_TURNS,
            "history": conversation_history,
            "direct_mode": direct_mode,
        }
        logger_obj.save(final_state=final_state)


if __name__ == "__main__":
    main()