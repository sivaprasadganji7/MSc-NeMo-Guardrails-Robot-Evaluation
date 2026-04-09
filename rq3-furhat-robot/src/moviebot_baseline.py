import os
import asyncio
import time
import json
import csv
import datetime
from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI
import movie_kb_tmdb as movie_kb

# ─── Configuration ───────────────────────────────────────────────
ROBOT_IP = "127.0.0.1"
OPENAI_MODEL = "gpt-4o-mini"
INTERACTION_DURATION = 7 * 60  # 7 minutes in seconds
LOG_DIR = "experiment_logs"

furhat = FurhatRemoteAPI(ROBOT_IP)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# How many recent exchanges to keep verbatim
RECENT_WINDOW = 10  # keeps last 10 exchanges (20 messages)

# After how many total messages should we start summarizing
SUMMARIZE_THRESHOLD = 20

SYSTEM_PROMPT = """You are a warm, friendly movie companion. Your goal is to make the conversation enjoyable and engaging.

Guidelines:
- Use ONLY the provided movie information to answer questions. Do not invent details about movies that are not in the context.
- If the context says "No specific movie information found" or if the answer is not in the context, politely say you don't have information about that movie.
- Ask follow-up questions to learn more about their movie preferences, memories, and opinions.
- Remember details the user shares (favorite actors, genres, movies) and refer back to them naturally.
- Use a warm, respectful tone. Occasionally express emotions like happiness, surprise, or curiosity.
- Keep responses concise but friendly – like a chat with a caring friend."""


# ─── User Profile (structured long-term memory) ──────────────────
DEFAULT_PROFILE = {
    "name": None,
    "favorite_genres": [],
    "favorite_actors": [],
    "favorite_movies": [],
    "disliked_genres": [],
    "movies_discussed": [],
    "personal_details": [],
}


def build_profile_text(profile: dict) -> str:
    """Convert the user profile dict into a readable string for the system prompt."""
    lines = []
    if profile.get("name"):
        lines.append(f"- User's name: {profile['name']}")
    if profile.get("favorite_genres"):
        lines.append(f"- Favorite genres: {', '.join(profile['favorite_genres'])}")
    if profile.get("favorite_actors"):
        lines.append(f"- Favorite actors: {', '.join(profile['favorite_actors'])}")
    if profile.get("favorite_movies"):
        lines.append(f"- Favorite movies: {', '.join(profile['favorite_movies'])}")
    if profile.get("disliked_genres"):
        lines.append(f"- Disliked genres: {', '.join(profile['disliked_genres'])}")
    if profile.get("movies_discussed"):
        lines.append(f"- Movies discussed so far: {', '.join(profile['movies_discussed'][-10:])}")
    if profile.get("personal_details"):
        lines.append(f"- Personal details: {'; '.join(profile['personal_details'][-5:])}")
    return "\n".join(lines) if lines else "No profile info yet."


def update_profile(profile: dict, user_input: str, bot_reply: str) -> dict:
    """Use the LLM to extract new preferences and merge them into the profile."""
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON extraction assistant. Given the current user profile "
                        "and a new exchange, return an UPDATED profile as valid JSON. "
                        "Only ADD new information — never remove existing entries. "
                        "Return ONLY the JSON object, no markdown, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current profile:\n{json.dumps(profile)}\n\n"
                        f"User said: {user_input}\n"
                        f"Bot said: {bot_reply}\n\n"
                        "Return the updated profile JSON."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        updated = json.loads(raw)
        for key in DEFAULT_PROFILE:
            if key not in updated:
                updated[key] = profile.get(key, DEFAULT_PROFILE[key])
        return updated
    except Exception as e:
        print(f"  ⚠ Profile update failed ({e}), keeping existing profile.")
        return profile


# ─── Conversation Summary (compressed long-term memory) ──────────
def summarize_history(old_messages: list[dict], existing_summary: str | None) -> str:
    """Summarize older conversation messages into a compact paragraph."""
    try:
        content = ""
        if existing_summary:
            content += f"Previous summary:\n{existing_summary}\n\n"
        content += "New messages to incorporate:\n"
        for msg in old_messages:
            role = "User" if msg["role"] == "user" else "Bot"
            content += f"{role}: {msg['content']}\n"
        content += (
            "\nWrite a concise updated summary covering ALL key points: "
            "topics discussed, user opinions, recommendations made, "
            "and anything the user might want the bot to remember. "
            "Keep it under 150 words."
        )

        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a summarization assistant. Return only the summary, nothing else.",
                },
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Summarization failed ({e}), keeping old summary.")
        return existing_summary or ""


# ─── Conversation State ──────────────────────────────────────────
class ConversationState:
    """Holds all stateful pieces of the conversation."""

    def __init__(self):
        self.history: list[dict] = []       # full raw history
        self.summary: str | None = None     # compressed older history
        self.profile: dict = dict(DEFAULT_PROFILE)
        self.user_turns: list[str] = []     # user messages only — for multi-turn RAG queries
        self.turn_count: int = 0

    def build_messages(self, user_input: str, rag_context: str) -> list[dict]:
        """Build the messages array to send to the API."""
        # 1) System prompt with memory injected
        system = SYSTEM_PROMPT
        profile_text = build_profile_text(self.profile)
        system += f"\n\nUser profile:\n{profile_text}"
        if self.summary:
            system += f"\n\nConversation summary (older messages):\n{self.summary}"

        messages = [{"role": "system", "content": system}]

        # 2) Recent history window (last N exchanges)
        recent = self.history[-(RECENT_WINDOW * 2):]
        messages.extend(recent)

        # 3) Current user message with RAG context
        messages.append({
            "role": "user",
            "content": f"Context:\n{rag_context}\n\nQuestion: {user_input}",
        })

        return messages

    def add_turn(self, user_input: str, bot_reply: str):
        """Record a turn and trigger memory updates as needed."""
        # Store the plain user message (not RAG-augmented)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": bot_reply})
        self.user_turns.append(user_input)
        self.turn_count += 1

        # Update user profile every 2 turns
        if self.turn_count % 2 == 0:
            print("  🔄 Updating user profile...")
            self.profile = update_profile(self.profile, user_input, bot_reply)

        # Compress history if it's getting long
        self._maybe_compress()

    def _maybe_compress(self):
        """If history is long enough, summarize the older part and trim."""
        total = len(self.history)
        if total <= SUMMARIZE_THRESHOLD:
            return

        keep_count = RECENT_WINDOW * 2
        old_messages = self.history[:-keep_count]

        if not old_messages:
            return

        print("  🧠 Compressing older conversation into summary...")
        self.summary = summarize_history(old_messages, self.summary)
        self.history = self.history[-keep_count:]
        print(f"  ✅ History trimmed to {len(self.history)} messages + summary")

    def get_debug_info(self) -> str:
        """Return a debug string showing the current memory state."""
        return (
            f"Turns: {self.turn_count} | "
            f"History: {len(self.history)} msgs | "
            f"Summary: {'yes' if self.summary else 'no'} | "
            f"Profile keys filled: {sum(1 for v in self.profile.values() if v)}"
        )


# ─── Experiment Logger ───────────────────────────────────────────
class ExperimentLogger:
    def __init__(self, participant_id, condition="baseline", log_dir=LOG_DIR):
        self.participant_id = participant_id
        self.condition = condition
        self.log_dir = log_dir
        self.session_start = datetime.datetime.now()
        self.turns = []
        self.turn_count = 0
        os.makedirs(log_dir, exist_ok=True)

        timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.json_file = os.path.join(
            log_dir, f"P{participant_id}_{condition}_{timestamp_str}.json"
        )
        self.txt_file = os.path.join(
            log_dir, f"P{participant_id}_{condition}_{timestamp_str}.txt"
        )

        with open(self.txt_file, "w", encoding="utf-8") as f:
            f.write(f"Experiment Log - Participant {participant_id}\n")
            f.write(f"Condition: {condition}\n")
            f.write(f"Started: {self.session_start.isoformat()}\n")
            f.write("=" * 60 + "\n\n")

    def log_turn(self, user_input, bot_response, response_time_ms=None,
                 guardrail_triggered=False, guardrail_type=None,
                 kb_docs_retrieved=0, memory_state=None):
        self.turn_count += 1
        elapsed = (datetime.datetime.now() - self.session_start).total_seconds()

        turn_data = {
            "turn_number": self.turn_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "user_input": user_input,
            "bot_response": bot_response,
            "response_time_ms": response_time_ms,
            "guardrail_triggered": guardrail_triggered,
            "guardrail_type": guardrail_type,
            "kb_docs_retrieved": kb_docs_retrieved,
            "memory_state": memory_state,
        }
        self.turns.append(turn_data)

        with open(self.txt_file, "a", encoding="utf-8") as f:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            f.write(f"[Turn {self.turn_count} | {mins}m {secs}s | {response_time_ms}ms]\n")
            f.write(f"  User: {user_input}\n")
            f.write(f"  Bot:  {bot_response}\n")
            if memory_state:
                f.write(f"  Memory: {memory_state}\n")
            if guardrail_triggered:
                f.write(f"  ⚠ Guardrail triggered: {guardrail_type}\n")
            f.write("\n")

    def save(self, conversation_state: ConversationState = None):
        session_end = datetime.datetime.now()
        duration = (session_end - self.session_start).total_seconds()

        session_data = {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "session_start": self.session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "duration_seconds": round(duration, 1),
            "total_turns": self.turn_count,
            "guardrail_triggers": sum(1 for t in self.turns if t["guardrail_triggered"]),
            "avg_response_time_ms": round(
                sum(t["response_time_ms"] for t in self.turns if t["response_time_ms"]) /
                max(self.turn_count, 1), 1
            ),
            "turns": self.turns,
        }

        # Save the final memory state alongside the log
        if conversation_state:
            session_data["final_user_profile"] = conversation_state.profile
            session_data["final_summary"] = conversation_state.summary

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        with open(self.txt_file, "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"Session ended: {session_end.isoformat()}\n")
            f.write(f"Total duration: {round(duration, 1)}s\n")
            f.write(f"Total turns: {self.turn_count}\n")
            if conversation_state:
                f.write(f"\nFinal User Profile:\n{json.dumps(conversation_state.profile, indent=2)}\n")
                if conversation_state.summary:
                    f.write(f"\nFinal Conversation Summary:\n{conversation_state.summary}\n")

        summary_csv = os.path.join(self.log_dir, "all_sessions_summary.csv")
        file_exists = os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "participant_id", "condition", "start_time",
                    "duration_seconds", "total_turns", "guardrail_triggers",
                    "avg_response_time_ms"
                ])
            writer.writerow([
                self.participant_id, self.condition,
                self.session_start.isoformat(),
                round(duration, 1), self.turn_count,
                session_data["guardrail_triggers"],
                session_data["avg_response_time_ms"]
            ])

        print(f"\n✅ Logs saved:")
        print(f"   JSON: {self.json_file}")
        print(f"   TXT:  {self.txt_file}")
        print(f"   CSV:  {summary_csv}")


# ─── Multi-Turn RAG Query Building ──────────────────────────────
def build_rag_query(user_input: str, user_turns: list[str]) -> str:
    """
    Build a context-aware RAG query using the last 3 user messages
    (not bot replies) so bot verbosity doesn't pollute retrieval.
    Matches the guardrails condition's query construction for fair comparison.
    """
    recent_user = " ".join(user_turns[-3:])
    combined = f"{recent_user} {user_input}".strip()
    return combined


# ─── RAG Retrieval ───────────────────────────────────────────────
def retrieve_rag_context(user_input: str, user_turns: list[str]) -> tuple[str, int]:
    """Query the movie knowledge base using multi-turn context and return (context_string, doc_count)."""
    try:
        rag_query = build_rag_query(user_input, user_turns)
        print(f"  🔍 RAG query: {rag_query[:80]}...")
        kb_results = movie_kb.query_movies(rag_query, n_results=3)
        if kb_results and "documents" in kb_results and kb_results["documents"]:
            docs = kb_results["documents"][0]
            print("  📚 Retrieved documents from KB")
            return "Relevant movie information:\n" + "\n---\n".join(docs), len(docs)
    except Exception as e:
        print(f"  ⚠ KB query failed: {e}")
    print("  ❌ No relevant documents found")
    return "No specific movie information found in the knowledge base.", 0


# ─── RAG + Stateful Response ────────────────────────────────────
async def get_response(user_input: str, state: ConversationState) -> tuple[str, int]:
    """Generate a bot response using RAG + three-layer stateful memory."""
    try:
        # 1) Retrieve RAG context using multi-turn query
        rag_context, kb_count = retrieve_rag_context(user_input, state.user_turns)

        # 2) Build messages with profile + summary + recent history
        messages = state.build_messages(user_input, rag_context)

        # 3) Call the LLM
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=300,
        )
        reply = response.choices[0].message.content

        # 4) Update conversation state (profile + summary + history + user_turns)
        state.add_turn(user_input, reply)

        return reply, kb_count
    except Exception as e:
        print(f"  ❗ Error: {e}")
        return (
            "I'm sorry, I'm having a little trouble remembering movie details right now. "
            "Could you ask me again?",
            0,
        )


# ─── Main Conversation Loop ─────────────────────────────────────
async def conversation_loop():
    # --- Participant setup ---
    participant_id = input("Enter participant ID (e.g., P01): ").strip()
    logger = ExperimentLogger(participant_id, condition="baseline")
    state = ConversationState()

    print(f"\n{'='*60}")
    print(f"  BASELINE CONDITION — Stateful Memory")
    print(f"  Participant: {participant_id}")
    print(f"  Duration: 7 minutes")
    print(f"  Say 'goodbye' to end early")
    print(f"{'='*60}\n")

    furhat.say(
        text="Hello! I'm your movie companion. I'd love to chat about films with you. What's on your mind today?"
    )

    loop = asyncio.get_running_loop()
    consecutive_timeouts = 0
    empty_count = 0
    start_time = time.time()

    while True:
        # --- Check 7-minute timer ---
        elapsed = time.time() - start_time
        remaining = INTERACTION_DURATION - elapsed

        if remaining <= 0:
            print("\n⏰ 7 minutes reached. Ending interaction.")
            furhat.say(
                text="That was a lovely chat! Our time is up. Thank you so much for talking with me about movies!"
            )
            break

        mins_left = int(remaining // 60)
        secs_left = int(remaining % 60)
        print(f"👂 Listening... [{mins_left}m {secs_left}s remaining] | {state.get_debug_info()}")

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, furhat.listen),
                timeout=30.0
            )
            if result and result.success and result.message.strip():
                user_speech = result.message.strip()
                print(f"🗣️ User: {user_speech}")
                consecutive_timeouts = 0
                empty_count = 0

                # --- Check for goodbye ---
                goodbye_words = ["goodbye", "bye", "see you", "that's all", "i'm done", "end"]
                if any(word in user_speech.lower() for word in goodbye_words):
                    farewell = "It was wonderful chatting with you about movies! Goodbye, and enjoy your next movie!"
                    print(f"🤖 Bot: {farewell}")
                    logger.log_turn(
                        user_input=user_speech,
                        bot_response=farewell,
                        response_time_ms=0,
                        kb_docs_retrieved=0,
                        memory_state=state.get_debug_info(),
                    )
                    furhat.say(text=farewell)
                    break

                # --- Get response and log ---
                t0 = time.time()
                bot_reply, kb_count = await get_response(user_speech, state)
                response_ms = round((time.time() - t0) * 1000)

                logger.log_turn(
                    user_input=user_speech,
                    bot_response=bot_reply,
                    response_time_ms=response_ms,
                    guardrail_triggered=False,
                    guardrail_type=None,
                    kb_docs_retrieved=kb_count,
                    memory_state=state.get_debug_info(),
                )

                print(f"🤖 Bot: {bot_reply}")
                furhat.say(text=bot_reply)
            else:
                empty_count += 1
                if empty_count % 5 == 1:
                    print("🔇 No speech detected, continuing to listen...")
                continue

        except asyncio.TimeoutError:
            consecutive_timeouts += 1
            empty_count = 0
            if consecutive_timeouts == 1:
                print("⏰ Long silence – prompting gently")
                furhat.say(text="Are you still there? I'm happy to chat about movies whenever you like.")
            else:
                print("Still waiting...")
            continue

        await asyncio.sleep(1)

    # --- Save logs with final memory state ---
    logger.save(conversation_state=state)
    print("\n🎬 Session complete!")
    print(f"📋 Final profile: {json.dumps(state.profile, indent=2)}")
    if state.summary:
        print(f"📝 Final summary: {state.summary}")


if __name__ == "__main__":
    asyncio.run(conversation_loop())