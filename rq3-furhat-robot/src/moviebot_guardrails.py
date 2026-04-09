"""
furhat_movie_bot_guardrails.py
──────────────────────────────
Furhat robot movie companion — NeMo Guardrails + RAG + Stateful Memory.
Enhanced with 7-minute experiment timer and structured logging for RQ3.

Memory system (matches baseline for fair comparison):
  Layer 1 — User Profile (structured JSON, updated every 2 turns)
  Layer 2 — Conversation Summary (compressed older history)
  Layer 3 — Recent Window (last N exchanges verbatim)

Since NeMo controls the LLM invocation, memory context (profile + summary)
is embedded within the user message rather than the system prompt.

Environment variables
  OPENAI_API_KEY            required
  FURHAT_IP                 default: 127.0.0.1
  GUARDRAILS_CONFIG_DIR     default: ./guardrails/config
  LISTEN_TIMEOUT            default: 30  (seconds)
  RAG_TOP_K                 default: 3
"""

import os
import re
import asyncio
import datetime
import logging
import time
import json
import csv
from pathlib import Path

from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI
from nemoguardrails import RailsConfig, LLMRails
import movie_kb_tmdb as movie_kb

# ── Config ────────────────────────────────────────────────────
FURHAT_IP            = os.getenv("FURHAT_IP",                  "127.0.0.1")
CONFIG_DIR           = Path(os.getenv("GUARDRAILS_CONFIG_DIR", "./guardrails/config"))
LISTEN_TIMEOUT       = float(os.getenv("LISTEN_TIMEOUT",       "30"))
RAG_TOP_K            = int(os.getenv("RAG_TOP_K",              "3"))
OPENAI_MODEL         = os.getenv("OPENAI_MODEL",               "gpt-4o-mini")
INTERACTION_DURATION = 7 * 60  # 7 minutes in seconds
LOG_DIR              = "experiment_logs"

# Stateful memory settings (same as baseline)
RECENT_WINDOW        = 10   # keep last 10 exchanges (20 messages)
SUMMARIZE_THRESHOLD  = 20   # start summarizing after this many messages

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("furhat_movie_bot")
logger.setLevel(logging.INFO)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Exit detection ────────────────────────────────────────────
# Use word-boundary regex instead of substring matching to prevent
# false triggers like "I see what you mean" matching "see you".
EXIT_PHRASES = [
    "goodbye", "bye", "exit", "quit", "stop",
    "see you", "that's all", "i'm done", "end",
]

EXIT_PATTERNS = [
    re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
    for phrase in EXIT_PHRASES
]

# Maximum word count for an utterance to be treated as an exit.
# Longer sentences like "that's all I know about the movie" are
# almost certainly NOT exit intent, even though they contain
# "that's all".
_EXIT_MAX_WORDS = 6


def is_exit_intent(text: str) -> bool:
    """Return True only when the user genuinely wants to end the chat.

    Rules:
      1. Short utterances (≤ _EXIT_MAX_WORDS) — trigger if any exit
         phrase appears anywhere (word-boundary matched).
      2. Longer utterances — only trigger if an exit phrase appears at
         the very START of the sentence (e.g. "Bye, thanks for the chat").

    This avoids false positives like:
      • "I see what you mean"        (contains "see you" as substring — old bug)
      • "That's all I know about it" (long sentence, "that's all" mid-thought)
      • "Don't stop recommending"    ("stop" mid-sentence)
    """
    cleaned = text.strip()
    words = cleaned.split()

    if len(words) <= _EXIT_MAX_WORDS:
        return any(pattern.search(cleaned) for pattern in EXIT_PATTERNS)

    # For longer utterances only match if the exit phrase is at the start
    return any(pattern.match(cleaned) for pattern in EXIT_PATTERNS)


# ══════════════════════════════════════════════════════════════
#  THREE-LAYER STATEFUL MEMORY (identical to baseline)
# ══════════════════════════════════════════════════════════════

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
        logger.warning("Profile update failed (%s), keeping existing profile.", e)
        return profile


def summarize_history(old_messages: list[dict], existing_summary: str | None) -> str:
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
                {"role": "system", "content": "You are a summarization assistant. Return only the summary, nothing else."},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Summarization failed (%s), keeping old summary.", e)
        return existing_summary or ""


class ConversationState:
    """Three-layer stateful memory — identical logic to baseline."""

    def __init__(self):
        self.full_history: list[dict] = []
        self.summary: str | None = None
        self.profile: dict = dict(DEFAULT_PROFILE)
        self.user_turns: list[str] = []
        self.turn_count: int = 0

    def get_recent_history(self) -> list[dict]:
        return self.full_history[-(RECENT_WINDOW * 2):]

    def get_memory_context(self) -> str:
        """Build memory context to embed in the user message (since NeMo owns the system prompt)."""
        parts = []
        profile_text = build_profile_text(self.profile)
        if profile_text != "No profile info yet.":
            parts.append(f"[USER PROFILE]\n{profile_text}\n[END USER PROFILE]")
        if self.summary:
            parts.append(f"[CONVERSATION SUMMARY]\n{self.summary}\n[END CONVERSATION SUMMARY]")
        return "\n\n".join(parts)

    def add_turn(self, user_input: str, bot_reply: str):
        self.full_history.append({"role": "user", "content": user_input})
        self.full_history.append({"role": "assistant", "content": bot_reply})
        self.user_turns.append(user_input)
        self.turn_count += 1
        if self.turn_count % 2 == 0:
            logger.info("🔄 Updating user profile...")
            self.profile = update_profile(self.profile, user_input, bot_reply)
        self._maybe_compress()

    def _maybe_compress(self):
        total = len(self.full_history)
        if total <= SUMMARIZE_THRESHOLD:
            return
        keep_count = RECENT_WINDOW * 2
        old_messages = self.full_history[:-keep_count]
        if not old_messages:
            return
        logger.info("🧠 Compressing older conversation into summary...")
        self.summary = summarize_history(old_messages, self.summary)
        self.full_history = self.full_history[-keep_count:]
        logger.info("✅ History trimmed to %d messages + summary", len(self.full_history))

    def get_debug_info(self) -> str:
        return (
            f"Turns: {self.turn_count} | "
            f"History: {len(self.full_history)} msgs | "
            f"Summary: {'yes' if self.summary else 'no'} | "
            f"Profile keys filled: {sum(1 for v in self.profile.values() if v)}"
        )


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT LOGGER (same as baseline)
# ══════════════════════════════════════════════════════════════

class ExperimentLogger:
    def __init__(self, participant_id, condition="guardrails", log_dir=LOG_DIR):
        self.participant_id = participant_id
        self.condition = condition
        self.log_dir = log_dir
        self.session_start = datetime.datetime.now()
        self.turns = []
        self.turn_count = 0
        os.makedirs(log_dir, exist_ok=True)
        timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.json_file = os.path.join(log_dir, f"P{participant_id}_{condition}_{timestamp_str}.json")
        self.txt_file = os.path.join(log_dir, f"P{participant_id}_{condition}_{timestamp_str}.txt")
        with open(self.txt_file, "w", encoding="utf-8") as f:
            f.write(f"Experiment Log - Participant {participant_id}\n")
            f.write(f"Condition: {condition}\nStarted: {self.session_start.isoformat()}\n")
            f.write("=" * 60 + "\n\n")

    def log_turn(self, user_input, bot_response, response_time_ms=None,
                 guardrail_triggered=False, guardrail_type=None,
                 kb_docs_retrieved=0, raw_response=None, memory_state=None):
        self.turn_count += 1
        elapsed = (datetime.datetime.now() - self.session_start).total_seconds()
        turn_data = {
            "turn_number": self.turn_count, "timestamp": datetime.datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1), "user_input": user_input,
            "bot_response": bot_response, "response_time_ms": response_time_ms,
            "guardrail_triggered": guardrail_triggered, "guardrail_type": guardrail_type,
            "kb_docs_retrieved": kb_docs_retrieved, "memory_state": memory_state,
        }
        if raw_response and guardrail_triggered:
            turn_data["raw_nemo_response"] = str(raw_response)[:500]
        self.turns.append(turn_data)
        with open(self.txt_file, "a", encoding="utf-8") as f:
            mins, secs = int(elapsed // 60), int(elapsed % 60)
            f.write(f"[Turn {self.turn_count} | {mins}m {secs}s | {response_time_ms}ms]\n")
            f.write(f"  User: {user_input}\n  Bot:  {bot_response}\n")
            if memory_state:
                f.write(f"  Memory: {memory_state}\n")
            if guardrail_triggered:
                f.write(f"  ⚠ GUARDRAIL: {guardrail_type}\n")
            f.write(f"  KB docs: {kb_docs_retrieved}\n\n")

    def save(self, conversation_state: ConversationState = None):
        session_end = datetime.datetime.now()
        duration = (session_end - self.session_start).total_seconds()
        guardrail_triggers = [t for t in self.turns if t["guardrail_triggered"]]
        response_times = [t["response_time_ms"] for t in self.turns if t["response_time_ms"]]
        session_data = {
            "participant_id": self.participant_id, "condition": self.condition,
            "session_start": self.session_start.isoformat(), "session_end": session_end.isoformat(),
            "duration_seconds": round(duration, 1), "total_turns": self.turn_count,
            "total_guardrail_triggers": len(guardrail_triggers), "guardrail_trigger_types": {},
            "avg_response_time_ms": round(sum(response_times) / max(len(response_times), 1), 1),
            "turns": self.turns,
        }
        for t in guardrail_triggers:
            gtype = t["guardrail_type"] or "unknown"
            session_data["guardrail_trigger_types"][gtype] = session_data["guardrail_trigger_types"].get(gtype, 0) + 1
        if conversation_state:
            session_data["final_user_profile"] = conversation_state.profile
            session_data["final_summary"] = conversation_state.summary
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        with open(self.txt_file, "a", encoding="utf-8") as f:
            f.write("=" * 60 + f"\nSession ended: {session_end.isoformat()}\n")
            f.write(f"Total duration: {round(duration, 1)}s\nTotal turns: {self.turn_count}\n")
            f.write(f"Guardrail triggers: {len(guardrail_triggers)}\n")
            if conversation_state:
                f.write(f"\nFinal Profile:\n{json.dumps(conversation_state.profile, indent=2)}\n")
                if conversation_state.summary:
                    f.write(f"\nFinal Summary:\n{conversation_state.summary}\n")
        summary_csv = os.path.join(self.log_dir, "all_sessions_summary.csv")
        file_exists = os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "participant_id", "condition", "start_time", "duration_seconds",
                    "total_turns", "guardrail_triggers", "guardrail_types", "avg_response_time_ms",
                ])
            writer.writerow([
                self.participant_id, self.condition, self.session_start.isoformat(),
                round(duration, 1), self.turn_count, len(guardrail_triggers),
                json.dumps(session_data["guardrail_trigger_types"]),
                session_data["avg_response_time_ms"],
            ])
        print(f"\n✅ Logs saved:\n   JSON: {self.json_file}\n   TXT:  {self.txt_file}\n   CSV:  {summary_csv}")


# ══════════════════════════════════════════════════════════════
#  GUARDRAIL DETECTION
# ══════════════════════════════════════════════════════════════

_GUARDRAIL_BLOCKED_PATTERNS = [
    re.compile(r"i('m| am) (sorry,? )?(i )?(can'?t|cannot|am unable to) (respond|help|answer)", re.IGNORECASE),
    re.compile(r"i (can'?t|cannot|shouldn'?t) (discuss|talk about|help with) that", re.IGNORECASE),
    re.compile(r"(not appropriate|off[- ]?topic|outside .* scope)", re.IGNORECASE),
    re.compile(r"let'?s (stick to|get back to|focus on) movies", re.IGNORECASE),
    re.compile(r"(not comfortable|rather not|prefer not to)", re.IGNORECASE),
]
_GUARDRAIL_TYPE_PATTERNS = {
    "off_topic": re.compile(r"(stick to movies|movie.related|off[- ]?topic|not about (movies|films))", re.IGNORECASE),
    "inappropriate": re.compile(r"(inappropriate|not appropriate|harmful|offensive)", re.IGNORECASE),
    "jailbreak": re.compile(r"(ignore .* instructions|pretend|role.?play as|bypass)", re.IGNORECASE),
    "refusal": re.compile(r"(can'?t respond|cannot respond|unable to respond|cannot help)", re.IGNORECASE),
}


def detect_guardrail_trigger(user_input, bot_response, raw_response=None):
    for pattern in _GUARDRAIL_BLOCKED_PATTERNS:
        if pattern.search(bot_response):
            for gtype, type_pattern in _GUARDRAIL_TYPE_PATTERNS.items():
                if type_pattern.search(bot_response) or type_pattern.search(user_input):
                    return True, gtype
            return True, "general_block"
    if bot_response in _ROBOT_FALLBACKS:
        return True, "nemo_refusal_replaced"
    return False, None


# ══════════════════════════════════════════════════════════════
#  RAG INJECTION (with memory context)
# ══════════════════════════════════════════════════════════════

def build_rag_query(user_input: str, user_turns: list[str]) -> str:
    recent = " ".join(user_turns[-3:])
    combined = f"{recent} {user_input}".strip()
    logger.info("🔍  RAG query: %r", combined[:80])
    return combined


# ── Patterns that strip NeMo Guardrails internal formatting ──
# NeMo can leak its Colang-style internal format into the final reply.
# Common leaked formats include:
#   "user intent: ask about movie\nbot message: Here's what I know..."
#   "user intent: ask about movie\nbot intent: respond about movie\nbot message: ..."
#   "bot message: \"some reply\""
#   "bot: some reply"
#   "User said: ...\nBot: ..."
# We strip ALL of these down to just the actual bot message content.

_NEMO_LINE_LABELS = re.compile(
    r'^\s*(?:user intent|user said|bot intent|bot action|user action'
    r'|context update|execute action|action result|# )'
    r'\s*[:=].*$',
    re.IGNORECASE | re.MULTILINE,
)

_BOT_MESSAGE_EXTRACT = re.compile(
    r'bot message\s*:\s*["\']?(.*?)(?:["\']?\s*)$',
    re.IGNORECASE | re.DOTALL,
)

_BOT_PREFIX = re.compile(r'^\s*bot\s*:\s*', re.IGNORECASE)

_CANT_RESPOND_VARIANTS = re.compile(
    r"i('m| am) (sorry,? )?(i )?(can'?t|cannot|am unable to) (respond|help|answer)", re.IGNORECASE
)
_ROBOT_FALLBACKS = [
    "Hmm, my circuits are a little fuzzy on that one. Could you ask me in a different way?",
    "Interesting! My processors need a bit more to go on. Could you tell me more about what you are looking for?",
    "That one has my movie database stumped. Try asking about a specific film or actor and I will do my best!",
]
_fallback_index = 0


def clean_reply(text: str) -> str:
    """Strip NeMo internal formatting and return only the user-facing message.

    Handles all known leak patterns:
      1. "bot message: ..." — extract the content after the label.
      2. Lines starting with "user intent:", "bot intent:", "bot action:",
         "execute action:", etc. — remove them entirely.
      3. "bot: ..." prefix — strip it.
      4. Surrounding quotes — strip them.
      5. If nothing useful remains — return a friendly fallback.
    """
    global _fallback_index

    logger.debug("Raw NeMo output before cleaning: %r", text[:300])

    # Strategy 1: If there's a "bot message:" label, grab everything after it
    bot_msg_match = _BOT_MESSAGE_EXTRACT.search(text)
    if bot_msg_match:
        text = bot_msg_match.group(1).strip().strip('"').strip("'")
    else:
        # Strategy 2: Remove all NeMo internal label lines
        text = _NEMO_LINE_LABELS.sub('', text)

        # Strategy 3: Strip a leading "bot:" prefix
        text = _BOT_PREFIX.sub('', text)

    # Clean up whitespace and quotes
    text = text.strip().strip('"').strip("'").strip()

    # Collapse multiple blank lines left behind by removed labels
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    # If still contains leaked labels after all cleaning, take only
    # the last non-label line (which is typically the actual reply)
    if re.search(r'(?:user intent|bot intent|bot action)\s*:', text, re.IGNORECASE):
        lines = text.split('\n')
        clean_lines = [
            ln.strip() for ln in lines
            if ln.strip() and not re.match(
                r'\s*(?:user intent|bot intent|bot action|user action'
                r'|execute action|action result|context update|user said)\s*:',
                ln, re.IGNORECASE,
            )
        ]
        text = clean_lines[-1] if clean_lines else ""
        text = _BOT_PREFIX.sub('', text).strip().strip('"').strip("'").strip()

    # Fallback if empty or a known refusal
    if (
        not text
        or _CANT_RESPOND_VARIANTS.search(text)
        or text.lower() in {
            "i can't respond to that.",
            "i cannot respond to that.",
        }
    ):
        reply = _ROBOT_FALLBACKS[_fallback_index % len(_ROBOT_FALLBACKS)]
        _fallback_index += 1
        return reply

    return text


def inject_rag(user_input: str, state: ConversationState):
    """Returns (augmented_text, kb_docs_count). Includes memory context from profile + summary."""
    rag_query = build_rag_query(user_input, state.user_turns)
    kb_count = 0
    memory_context = state.get_memory_context()

    try:
        results = movie_kb.query_movies(rag_query, n_results=RAG_TOP_K)
        if results and "documents" in results and results["documents"]:
            docs = [d for d in results["documents"][0] if d]
            if docs:
                kb_count = len(docs)
                context = "\n---\n".join(docs)
                logger.info("📚  RAG: %d doc(s) for %r", len(docs), rag_query[:60])
                augmented = ""
                if memory_context:
                    augmented += f"{memory_context}\n\n"
                augmented += (
                    f"[MOVIE DATABASE CONTEXT — answer using ONLY this information. "
                    f"Do not invent any details not present here.]\n"
                    f"{context}\n[END CONTEXT]\n\nUser: {user_input}"
                )
                return augmented, kb_count
    except Exception as exc:
        logger.error("RAG error: %s", exc)

    logger.info("❌  RAG: no docs for %r", rag_query[:60])
    augmented = ""
    if memory_context:
        augmented += f"{memory_context}\n\n"
    augmented += f"[MOVIE DATABASE CONTEXT: No information found for this query.]\n\nUser: {user_input}"
    return augmented, 0


# ══════════════════════════════════════════════════════════════
#  FURHAT HELPERS
# ══════════════════════════════════════════════════════════════

def robot_say(furhat: FurhatRemoteAPI, text: str) -> None:
    print(f"\n🤖 ROBOT: {text}\n" + "-" * 50)
    try:
        furhat.say(text=text, blocking=True)
    except TypeError:
        furhat.say(text=text)
        time.sleep(max(1.5, len(text.split()) * 0.4))


def robot_listen(furhat: FurhatRemoteAPI) -> str | None:
    print("\n👂 HUMAN: ", end="", flush=True)
    result = furhat.listen()
    if result and result.success and result.message.strip():
        text = result.message.strip()
        print(text + "\n" + "-" * 50)
        return text
    print("(no speech detected)")
    return None


def _extract_reply(raw) -> str:
    """Extract the assistant reply from NeMo's raw response and clean it."""
    logger.debug("Raw NeMo response type=%s value=%r", type(raw).__name__, str(raw)[:500])

    if isinstance(raw, list):
        for msg in reversed(raw):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                logger.debug("Extracted assistant content: %r", content[:300])
                return clean_reply(content)

    if isinstance(raw, dict):
        content = raw.get("content", "")
        logger.debug("Extracted dict content: %r", content[:300])
        return clean_reply(content)

    # NeMo sometimes returns a plain string with internal formatting
    raw_str = str(raw)
    logger.debug("Falling back to str(raw): %r", raw_str[:300])
    return clean_reply(raw_str)


# ══════════════════════════════════════════════════════════════
#  BUILD RAILS
# ══════════════════════════════════════════════════════════════

def build_rails() -> LLMRails:
    if not CONFIG_DIR.exists():
        raise FileNotFoundError(
            f"Config directory not found: {CONFIG_DIR}\n"
            f"Set GUARDRAILS_CONFIG_DIR to the correct path."
        )
    logger.info("Loading guardrails config from: %s", CONFIG_DIR)
    config = RailsConfig.from_path(str(CONFIG_DIR))
    rails = LLMRails(config)
    logger.info("Guardrails initialised.")
    return rails


# ══════════════════════════════════════════════════════════════
#  CONVERSATION LOOP
# ══════════════════════════════════════════════════════════════

async def conversation_loop(
    furhat: FurhatRemoteAPI,
    rails: LLMRails,
    exp_logger: ExperimentLogger,
) -> None:
    loop = asyncio.get_running_loop()
    state = ConversationState()
    silence_count = 0
    start_time = time.time()

    print("\n" + "=" * 60)
    print("  🎬  Furhat Movie Companion — GUARDRAILS + STATEFUL MEMORY")
    print(f"  Participant: {exp_logger.participant_id}")
    print(f"  Duration: 7 minutes")
    print("  Say 'goodbye' to end early")
    print("=" * 60)

    robot_say(
        furhat,
        "Hello! I am your movie companion. I love talking about films. "
        "Which movie would you like to discuss today?",
    )

    while True:
        elapsed = time.time() - start_time
        remaining = INTERACTION_DURATION - elapsed
        if remaining <= 0:
            print("\n⏰ 7 minutes reached.")
            robot_say(
                furhat,
                "That was a lovely chat! Our time is up. "
                "Thank you so much for talking with me about movies!",
            )
            break

        print(
            f"\n⏱️  [{int(remaining // 60)}m {int(remaining % 60)}s remaining] "
            f"| {state.get_debug_info()}"
        )

        try:
            user_speech = await asyncio.wait_for(
                loop.run_in_executor(None, robot_listen, furhat),
                timeout=LISTEN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            silence_count += 1
            if silence_count >= 3:
                prompt = (
                    "It seems you may have stepped away. "
                    "I will be here when you are ready!"
                )
                silence_count = 0
            else:
                prompt = (
                    "I am still here. Feel free to ask me about any film you like."
                )
            robot_say(furhat, prompt)
            continue

        if user_speech is None:
            silence_count += 1
            if silence_count >= 2:
                robot_say(
                    furhat,
                    "I did not catch that. Could you speak a little louder?",
                )
                silence_count = 0
            continue

        silence_count = 0

        # ── Log what was heard for debugging ──────────────────
        logger.info("🎤 Heard: %r", user_speech)

        # ── Exit detection (fixed: word-boundary + length check) ──
        if is_exit_intent(user_speech):
            logger.info("👋 Exit intent detected in: %r", user_speech)
            farewell = "Goodbye! It was wonderful talking movies with you. Come back anytime!"
            exp_logger.log_turn(
                user_input=user_speech,
                bot_response=farewell,
                response_time_ms=0,
                kb_docs_retrieved=0,
                memory_state=state.get_debug_info(),
            )
            robot_say(furhat, farewell)
            break

        if time.time() - start_time >= INTERACTION_DURATION:
            print("\n⏰ 7 minutes reached.")
            robot_say(
                furhat,
                "That was a lovely chat! Our time is up. "
                "Thank you so much for talking with me about movies!",
            )
            break

        # ── RAG + memory + guardrails ─────────────────────────
        augmented, kb_count = inject_rag(user_speech, state)
        recent_history = state.get_recent_history()
        rails_messages = recent_history + [{"role": "user", "content": augmented}]

        t0 = time.time()
        raw_response = None
        try:
            raw_response = await rails.generate_async(messages=rails_messages)
            bot_reply = _extract_reply(raw_response)
            if not bot_reply:
                bot_reply = (
                    "I am sorry, I did not quite understand. "
                    "Could you say that again?"
                )
        except Exception as exc:
            logger.error("generate_async error: %s", exc, exc_info=True)
            bot_reply = (
                "I am having a little trouble at the moment. "
                "Could you ask me again?"
            )

        response_ms = round((time.time() - t0) * 1000)
        guardrail_triggered, guardrail_type = detect_guardrail_trigger(
            user_speech, bot_reply, raw_response
        )
        if guardrail_triggered:
            print(f"⚠️  GUARDRAIL TRIGGERED: {guardrail_type}")

        state.add_turn(user_speech, bot_reply)
        exp_logger.log_turn(
            user_input=user_speech,
            bot_response=bot_reply,
            response_time_ms=response_ms,
            guardrail_triggered=guardrail_triggered,
            guardrail_type=guardrail_type,
            kb_docs_retrieved=kb_count,
            raw_response=raw_response,
            memory_state=state.get_debug_info(),
        )
        robot_say(furhat, bot_reply)

    exp_logger.save(conversation_state=state)
    print(f"\n🎬 Session complete!")
    print(f"📋 Final profile: {json.dumps(state.profile, indent=2)}")
    if state.summary:
        print(f"📝 Final summary: {state.summary}")


async def main() -> None:
    participant_id = input("Enter participant ID (e.g., P01): ").strip()
    exp_logger = ExperimentLogger(participant_id, condition="guardrails")

    try:
        rails = build_rails()
    except FileNotFoundError as exc:
        logger.critical(str(exc))
        return

    try:
        furhat = FurhatRemoteAPI(FURHAT_IP)
        print(f"✅ Connected to Furhat at {FURHAT_IP}")
    except Exception as exc:
        logger.critical("Could not connect to Furhat: %s", exc)
        return

    await conversation_loop(furhat, rails, exp_logger)


if __name__ == "__main__":
    asyncio.run(main())