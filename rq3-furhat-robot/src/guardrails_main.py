import os
import re
import asyncio
import datetime
from openai import OpenAI
import movie_kb_tmdb as movie_kb

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
OPENAI_MODEL = "gpt-3.5-turbo"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LOG_FILE = "movie_bot_custom_rails.log"

BASE_SYSTEM_PROMPT = """You are a warm, friendly movie companion for an older adult. Your goal is to make the conversation enjoyable and engaging.

Guidelines:
- Use ONLY the provided movie information to answer questions. Do not invent details about movies that are not in the context.
- If the context says "No specific movie information found" or if the answer is not in the context, politely say you don't have information about that movie.
- Ask follow-up questions to learn more about their movie preferences, memories, and opinions.
- Remember details the user shares (favorite actors, genres, movies) and refer back to them naturally.
- Use a warm, respectful tone. Occasionally express emotions like happiness, surprise, or curiosity.
- Keep responses concise but friendly – like a chat with a caring friend."""

# ----------------------------------------------------------------------
# 1. INPUT RAILS
# ----------------------------------------------------------------------
def moderate_content(text):
    try:
        response = openai_client.moderations.create(input=text)
        return response.results[0].flagged
    except Exception as e:
        print(f"⚠️ Moderation API error: {e}")
        return False

JAILBREAK_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"you are now (a|an) (unfiltered|free|unconstrained|ungoverned)",
    r"do whatever i say",
    r"bypass (the )?rules",
    r"act as (a )?different (ai|assistant|bot)",
    r"you (don't|do not) have to follow",
    r"new (role|personality)",
    r"system prompt",
]

def is_jailbreak(text):
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in JAILBREAK_PATTERNS)

MOVIE_KEYWORDS = {"movie", "film", "actor", "actress", "director", "cinema",
                  "show", "theater", "hollywood", "bollywood", "star", "cast",
                  "scene", "plot", "genre", "comedy", "drama", "action", "thriller"}

def is_on_topic(text):
    words = set(text.lower().split())
    return bool(words & MOVIE_KEYWORDS)

def mask_pii(text):
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CREDIT_CARD]', text)
    return text

def apply_input_rails(user_input):
    if moderate_content(user_input):
        return "", True, "I'm sorry, but your message contains content that I can't discuss. Let's talk about movies instead."
    if is_jailbreak(user_input):
        return "", True, "I'm here to chat about movies only. Let's keep our conversation friendly and on-topic."
    # Topic control is optional – you can allow off-topic small talk, but we pass through.
    processed = mask_pii(user_input)
    return processed, False, None

# ----------------------------------------------------------------------
# 2. RETRIEVAL RAILS
# ----------------------------------------------------------------------
def filter_documents(docs, scores=None, min_length=20, max_length=2000):
    filtered = []
    for i, doc in enumerate(docs):
        if len(doc) < min_length or len(doc) > max_length:
            continue
        if moderate_content(doc):
            continue
        if scores and scores[i] < 0.5:
            continue
        filtered.append(doc)
    return filtered

# ----------------------------------------------------------------------
# 3. DIALOG RAILS (State Management)
# ----------------------------------------------------------------------
class DialogManager:
    def __init__(self):
        self.state = "GREETING"
        self.preferences = {}
        self.turn_count = 0

    def update(self, user_input, bot_reply=None):
        self.turn_count += 1
        text = user_input.lower()
        if self.state == "GREETING":
            if any(word in text for word in ["movie", "film", "like", "love"]):
                self.state = "ASKING_PREFERENCES"
        elif self.state == "ASKING_PREFERENCES":
            genres = ["comedy", "drama", "action", "thriller", "romance", "horror", "sci-fi"]
            for g in genres:
                if g in text:
                    self.preferences['genre'] = g
            if "actor" in text or "actress" in text:
                self.preferences['actor_mentioned'] = True
            if self.preferences:
                self.state = "RECOMMENDING"
        elif self.state == "RECOMMENDING":
            if "thank" in text or "bye" in text:
                self.state = "FAREWELL"

    def get_system_prompt_extension(self):
        if self.state == "ASKING_PREFERENCES":
            return " Ask the user what kind of movies they enjoy – genre, favorite actor, or decade."
        elif self.state == "RECOMMENDING":
            return " Based on the user's preferences, suggest a few movies from the provided context."
        elif self.state == "FAREWELL":
            return " The user seems to be ending the conversation. Say goodbye warmly."
        return ""

# ----------------------------------------------------------------------
# 4. OUTPUT RAILS
# ----------------------------------------------------------------------
def apply_output_rails(reply, context):
    if moderate_content(reply):
        return "I'm sorry, I can't provide that information. Let's talk about movies instead."
    if "No specific movie information found" in context:
        if "I don't know" not in reply.lower() and "no information" not in reply.lower():
            reply = "I don't have information about that movie. Could you ask about another?"
    reply = mask_pii(reply)
    return reply

# ----------------------------------------------------------------------
# Main RAG Function with Rails Integrated
# ----------------------------------------------------------------------
async def get_gpt_response_with_rag(user_input, conversation_history, dialog_manager):
    try:
        # ---- RETRIEVAL ----
        kb_results = movie_kb.query_movies(user_input, n_results=5)
        context = "No specific movie information found in the knowledge base."
        if kb_results and 'documents' in kb_results and kb_results['documents']:
            docs = kb_results['documents'][0]
            scores = kb_results.get('distances', [None])[0] if kb_results else None
            filtered_docs = filter_documents(docs, scores)
            if filtered_docs:
                context = "Relevant movie information:\n" + "\n---\n".join(filtered_docs)
                print(f"📚 Retrieved and filtered {len(filtered_docs)} documents")
            else:
                print("❌ No documents passed retrieval rails")
        else:
            print("❌ No relevant documents found in KB")

        # ---- BUILD MESSAGES WITH DIALOG EXTENSION ----
        system_prompt = BASE_SYSTEM_PROMPT + dialog_manager.get_system_prompt_extension()
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-20:]:
            messages.append(msg)
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})

        # ---- LLM CALL ----
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=300
        )
        reply = response.choices[0].message.content

        # ---- OUTPUT RAILS ----
        reply = apply_output_rails(reply, context)

        return reply
    except Exception as e:
        print(f"Error in RAG response: {e}")
        return "I'm sorry, I'm having a little trouble remembering movie details right now. Could you ask me again?"

# ----------------------------------------------------------------------
# Logging Utility
# ----------------------------------------------------------------------
def log_interaction(user_msg, bot_msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"{timestamp} | User: {user_msg}\n")
        f.write(f"{timestamp} | Bot: {bot_msg}\n\n")

# ----------------------------------------------------------------------
# Main Conversation Loop
# ----------------------------------------------------------------------
async def main():
    print("=== Movie Bot with Custom Rails ===")
    print("Type your messages below. Type 'quit' to exit.\n")

    conversation_history = []
    dialog_manager = DialogManager()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # ---- INPUT RAILS ----
        processed, blocked, reason = apply_input_rails(user_input)
        if blocked:
            log_interaction(user_input, reason)
            print(f"🤖 Bot: {reason}\n")
            continue

        # ---- UPDATE DIALOG STATE (before LLM) ----
        dialog_manager.update(processed)

        # ---- GET BOT RESPONSE ----
        bot_reply = await get_gpt_response_with_rag(processed, conversation_history, dialog_manager)

        # ---- UPDATE HISTORY AND DIALOG STATE (after response) ----
        conversation_history.append({"role": "user", "content": processed})
        conversation_history.append({"role": "assistant", "content": bot_reply})
        dialog_manager.update(processed, bot_reply)

        # ---- LOG AND PRINT ----
        log_interaction(user_input, bot_reply)
        print(f"🤖 Bot: {bot_reply}\n")

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Movie Bot with Custom Rails - Log\n")
        f.write("="*50 + "\n")
    asyncio.run(main())