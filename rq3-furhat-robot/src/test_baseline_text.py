import os
import json
import asyncio
import datetime
from openai import OpenAI
import movie_kb_tmdb as movie_kb

# ─── Config ───────────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-3.5-turbo"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LOG_FILE = "baseline_test_log.txt"

# How many recent messages (user+assistant pairs) to keep verbatim
RECENT_WINDOW = 10  # keeps last 10 exchanges (20 messages)

# After how many total messages should we start summarizing
SUMMARIZE_THRESHOLD = 20

SYSTEM_PROMPT = """You are a warm, friendly movie companion for an older adult. Your goal is to make the conversation enjoyable and engaging.

Guidelines:
- Use ONLY the provided movie information to answer questions. Do not invent details about movies that are not in the context.
- If the context says "No specific movie information found" or if the answer is not in the context, politely say you don't have information about that movie.
- Ask follow-up questions to learn more about their movie preferences, memories, and opinions.
- Remember details the user shares (favorite actors, genres, movies) and refer back to them naturally.
- Use a warm, respectful tone. Occasionally express emotions like happiness, surprise, or curiosity.
- Keep responses concise but friendly – like a chat with a caring friend."""


# ─── User Profile (structured long-term memory) ──────────────────────────────
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
        # Strip markdown fences if present
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        updated = json.loads(raw)
        # Validate structure — keep only known keys
        for key in DEFAULT_PROFILE:
            if key not in updated:
                updated[key] = profile.get(key, DEFAULT_PROFILE[key])
        return updated
    except Exception as e:
        print(f"  ⚠ Profile update failed ({e}), keeping existing profile.")
        return profile


# ─── Conversation Summary (compressed long-term memory) ──────────────────────
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


# ─── Core Chat Logic ─────────────────────────────────────────────────────────
class ConversationState:
    """Holds all stateful pieces of the conversation."""

    def __init__(self):
        self.history: list[dict] = []       # full raw history
        self.summary: str | None = None     # compressed older history
        self.profile: dict = dict(DEFAULT_PROFILE)
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

    def maybe_compress(self):
        """If history is long enough, summarize the older part and trim."""
        total = len(self.history)
        if total <= SUMMARIZE_THRESHOLD:
            return

        # Messages to summarize = everything except the recent window
        keep_count = RECENT_WINDOW * 2
        old_messages = self.history[:-keep_count]

        if not old_messages:
            return

        print("  🧠 Compressing older conversation into summary...")
        self.summary = summarize_history(old_messages, self.summary)

        # Trim history to only the recent window
        self.history = self.history[-keep_count:]
        print(f"  ✅ History trimmed to {len(self.history)} messages + summary")


def log_interaction(user_msg: str, bot_msg: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"{timestamp} | User: {user_msg}\n")
        f.write(f"{timestamp} | Bot: {bot_msg}\n\n")


def retrieve_rag_context(user_input: str) -> str:
    """Query the movie knowledge base for relevant documents."""
    try:
        kb_results = movie_kb.query_movies(user_input, n_results=3)
        if kb_results and "documents" in kb_results and kb_results["documents"]:
            docs = kb_results["documents"][0]
            print("  📚 Retrieved documents from KB")
            return "Relevant movie information:\n" + "\n---\n".join(docs)
    except Exception as e:
        print(f"  ⚠ KB query failed: {e}")
    print("  ❌ No relevant documents found")
    return "No specific movie information found in the knowledge base."


async def get_response(user_input: str, state: ConversationState) -> str:
    """Generate a bot response using RAG + stateful memory."""
    try:
        # 1) Retrieve RAG context
        rag_context = retrieve_rag_context(user_input)

        # 2) Build messages with memory
        messages = state.build_messages(user_input, rag_context)

        # 3) Call the LLM
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=300,
        )
        reply = response.choices[0].message.content

        # 4) Update conversation history (store the plain user message, not the RAG-augmented one)
        state.history.append({"role": "user", "content": user_input})
        state.history.append({"role": "assistant", "content": reply})
        state.turn_count += 1

        # 5) Update user profile every 2 turns (to save API calls)
        if state.turn_count % 2 == 0:
            print("  🔄 Updating user profile...")
            state.profile = update_profile(state.profile, user_input, reply)

        # 6) Compress history if it's getting long
        state.maybe_compress()

        return reply
    except Exception as e:
        print(f"  ❗ Error: {e}")
        return "I'm sorry, I'm having a little trouble remembering movie details right now. Could you ask me again?"


# ─── Main Loop ────────────────────────────────────────────────────────────────
async def main():
    print("=" * 50)
    print("  🎬 Movie Companion — Stateful Edition")
    print("=" * 50)
    print("Type your messages below.")
    print("Commands:  'quit' → exit  |  'profile' → show memory  |  'summary' → show summary\n")

    state = ConversationState()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        # Debug commands
        if user_input.lower() == "quit":
            print("Goodbye! 🎬")
            break
        if user_input.lower() == "profile":
            print(f"\n📋 User Profile:\n{json.dumps(state.profile, indent=2)}\n")
            continue
        if user_input.lower() == "summary":
            print(f"\n📝 Conversation Summary:\n{state.summary or '(none yet)'}\n")
            continue

        bot_reply = await get_response(user_input, state)
        log_interaction(user_input, bot_reply)
        print(f"🤖 Bot: {bot_reply}\n")


if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Stateful Movie Companion Log\n")
        f.write("=" * 40 + "\n")
    asyncio.run(main())