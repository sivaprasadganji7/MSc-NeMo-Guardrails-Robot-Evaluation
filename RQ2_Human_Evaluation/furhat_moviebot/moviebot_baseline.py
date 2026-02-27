import os
import asyncio
import datetime
from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI
import movie_kb_tmdb as movie_kb

# Configuration
ROBOT_IP = "127.0.0.1"
OPENAI_MODEL = "gpt-3.5-turbo"

furhat = FurhatRemoteAPI(ROBOT_IP)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced system prompt for warm, empathetic conversation
SYSTEM_PROMPT = """You are a warm, friendly movie companion for an older adult. Your goal is to make the conversation enjoyable and engaging.

Guidelines:
- Use ONLY the provided movie information to answer questions. Do not invent details about movies that are not in the context.
- If the context says "No specific movie information found" or if the answer is not in the context, politely say you don't have information about that movie.
- Ask follow-up questions to learn more about their movie preferences, memories, and opinions.
- Remember details the user shares (favorite actors, genres, movies) and refer back to them naturally.
- Use a warm, respectful tone. Occasionally express emotions like happiness, surprise, or curiosity.
- Keep responses concise but friendly – like a chat with a caring friend."""

conversation_history = []
MAX_HISTORY = 10
LOG_FILE = "baseline_log.txt"

def log_interaction(user_msg, bot_msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"{timestamp} | User: {user_msg}\n")
        f.write(f"{timestamp} | Bot: {bot_msg}\n\n")

async def get_gpt_response_with_rag(user_input):
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
        for msg in conversation_history[-MAX_HISTORY*2:]:
            messages.append(msg)
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})
        
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=300
        )
        reply = response.choices[0].message.content
        
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": reply})
        if len(conversation_history) > MAX_HISTORY * 2:
            conversation_history[:2] = []
        
        return reply
    except Exception as e:
        print(f"Error in RAG response: {e}")
        return "I'm sorry, I'm having a little trouble remembering movie details right now. Could you ask me again?"

async def conversation_loop():
    furhat.say(text="Hello! I'm your movie companion. I'd love to chat about films with you. What's on your mind today?")
    
    loop = asyncio.get_running_loop()
    consecutive_timeouts = 0
    empty_count = 0
    
    while True:
        print("👂 Listening... (will wait up to 20 seconds)")
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, furhat.listen),
                timeout=20.0
            )
            if result and result.success and result.message.strip():
                user_speech = result.message.strip()
                print(f"🗣️ User: {user_speech}")
                consecutive_timeouts = 0
                empty_count = 0
                
                bot_reply = await get_gpt_response_with_rag(user_speech)
                log_interaction(user_speech, bot_reply)
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

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Baseline Interaction Log\n")
        f.write("="*40 + "\n")
    asyncio.run(conversation_loop())