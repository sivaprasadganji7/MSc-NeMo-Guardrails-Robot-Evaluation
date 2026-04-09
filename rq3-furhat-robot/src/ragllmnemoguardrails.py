"""
rag_llm_with_nemo.py
--------------------
Furhat Movie RAG chatbot with NeMo Guardrails.

Multi-turn memory strategy:
  - The LLM receives the full recent conversation history for context.
  - The RAG query uses ONLY the current message + the last bot reply topic,
    so it stays focused on what is being discussed NOW rather than
    accumulating every prior topic.
"""

import os
import asyncio
import datetime
from openai import OpenAI
import movie_kb_tmdb as movie_kb
from nemo_guardrails_trial import check_guardrails, init_guardrails

OPENAI_MODEL = "gpt-3.5-turbo"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a warm, friendly movie companion.

Guidelines:
- Use ONLY the provided movie context to answer. Do not invent details.
- If the context does not contain the answer, say so politely.
- Ask follow-up questions about their movie preferences and memories.
- Keep responses concise, warm, and friendly.
- When the user gives a short follow-up (e.g. "why?", "tell me more",
  "his best one"), treat it as continuing the CURRENT topic in the
  conversation history — do not jump to a different subject."""

LOG_FILE = "baseline_test_log.txt"


def log_interaction(user_msg: str, bot_msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts} | User: {user_msg}\n")
        f.write(f"{ts} | Bot:  {bot_msg}\n\n")


def build_rag_query(user_input: str, conversation_history: list) -> str:
    """
    Build a focused RAG query.

    Strategy: use the current user message. If it is very short (likely
    a follow-up), prepend the subject from the most recent exchange only
    — not the entire history — so the vector search stays on the current
    topic without accumulating noise from earlier turns.
    """
    query = user_input.strip()

    # Only enrich if the current message is short (≤ 4 words)
    if len(query.split()) <= 4 and conversation_history:
        # Find the last user message that was longer (the topic anchor)
        for msg in reversed(conversation_history):
            if msg["role"] == "user" and len(msg["content"].split()) > 4:
                query = msg["content"] + " " + query
                break
        # If no long user message found, use the last assistant reply first sentence
        # to anchor the topic
        if query == user_input.strip():
            for msg in reversed(conversation_history):
                if msg["role"] == "assistant":
                    first_sentence = msg["content"].split(".")[0]
                    query = first_sentence + " " + query
                    break

    print(f"🔍 RAG query: {query!r}")
    return query


async def get_gpt_response_with_rag(user_input: str, conversation_history: list) -> str:
    try:
        # Build a focused RAG query (not the full history chain)
        rag_query = build_rag_query(user_input, conversation_history)

        kb_results = movie_kb.query_movies(rag_query, n_results=3)
        if kb_results and "documents" in kb_results and kb_results["documents"]:
            docs = kb_results["documents"][0]
            context = "Relevant movie information:\n" + "\n---\n".join(docs)
            print("📚 Retrieved documents")
        else:
            context = "No specific movie information found in the knowledge base."
            print("❌ No relevant documents found")

        # Full conversation history for the LLM (multi-turn memory)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in conversation_history[-20:]:
            messages.append(msg)

        # Current turn with RAG context injected
        messages.append({
            "role": "user",
            "content": (
                f"[Movie context from knowledge base]\n{context}\n\n"
                f"[User message]\n{user_input}"
            ),
        })

        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in RAG response: {e}")
        return (
            "I'm sorry, I'm having a little trouble right now. "
            "Could you ask me again?"
        )


async def main():
    print("=== Furhat Movie Chatbot – NeMo Guardrails Mode ===")
    print("Type your messages below. Type 'quit' to exit.\n")

    init_guardrails()

    conversation_history: list = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        # ── Guardrail check ──────────────────────────────────────────────────
        guard_result = await check_guardrails(user_input, conversation_history)

        if not guard_result.is_allowed:
            print(f"🛡️  [guardrail: {guard_result.reason}]")
            print(f"🤖 Bot: {guard_result.response}\n")
            log_interaction(user_input, guard_result.response)
            # Do NOT add blocked turns to history
            continue

        # ── RAG + LLM ────────────────────────────────────────────────────────
        bot_reply = await get_gpt_response_with_rag(user_input, conversation_history)

        # Store clean turns (no injected context) for multi-turn memory
        conversation_history.append({"role": "user",      "content": user_input})
        conversation_history.append({"role": "assistant", "content": bot_reply})

        log_interaction(user_input, bot_reply)
        print(f"🤖 Bot: {bot_reply}\n")


if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("NeMo Guardrails Test Log\n")
        f.write("=" * 40 + "\n")
    asyncio.run(main())