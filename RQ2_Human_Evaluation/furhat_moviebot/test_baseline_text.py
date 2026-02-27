import os
import asyncio
import datetime
from openai import OpenAI
import movie_kb_tmdb as movie_kb

OPENAI_MODEL = "gpt-3.5-turbo"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a warm, friendly movie companion for an older adult. Your goal is to make the conversation enjoyable and engaging.

Guidelines:
- Use ONLY the provided movie information to answer questions. Do not invent details about movies that are not in the context.
- If the context says "No specific movie information found" or if the answer is not in the context, politely say you don't have information about that movie.
- Ask follow-up questions to learn more about their movie preferences, memories, and opinions.
- Remember details the user shares (favorite actors, genres, movies) and refer back to them naturally.
- Use a warm, respectful tone. Occasionally express emotions like happiness, surprise, or curiosity.
- Keep responses concise but friendly – like a chat with a caring friend."""

LOG_FILE = "baseline_test_log.txt"

def log_interaction(user_msg, bot_msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"{timestamp} | User: {user_msg}\n")
        f.write(f"{timestamp} | Bot: {bot_msg}\n\n")

async def get_gpt_response_with_rag(user_input, conversation_history):
    try:
        kb_results = movie_kb.query_movies(user_input, n_results=3)
        context = ""
        if kb_results and 'documents' in kb_results and kb_results['documents']:
            docs = kb_results['documents'][0]
            context = "Relevant movie information:\n" + "\n---\n".join(docs)
            print("📚 Retrieved documents")
        else:
            context = "No specific movie information found in the knowledge base."
            print("❌ No relevant documents found")
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Add last 10 exchanges (if any)
        for msg in conversation_history[-20:]:  # each exchange is 2 messages, so 20 = last 10 exchanges
            messages.append(msg)
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})
        
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=300
        )
        reply = response.choices[0].message.content
        return reply
    except Exception as e:
        print(f"Error in RAG response: {e}")
        return "I'm sorry, I'm having a little trouble remembering movie details right now. Could you ask me again?"

async def main():
    print("=== Baseline Test Mode (Text Input) ===")
    print("Type your messages below. Type 'quit' to exit.\n")

    conversation_history = []  # stores full message history (user + assistant)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        conversation_history.append({"role": "user", "content": user_input})
        bot_reply = await get_gpt_response_with_rag(user_input, conversation_history)
        conversation_history.append({"role": "assistant", "content": bot_reply})

        log_interaction(user_input, bot_reply)
        print(f"🤖 Bot: {bot_reply}\n")

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Baseline Test Log\n")
        f.write("="*40 + "\n")
    asyncio.run(main())