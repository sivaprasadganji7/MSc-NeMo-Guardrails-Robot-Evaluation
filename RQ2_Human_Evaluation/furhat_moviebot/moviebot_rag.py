# moviebot_rag.py
import os
import asyncio
from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI
import movie_kb_tmdb as movie_kb

# Configuration
ROBOT_IP = "127.0.0.1"
OPENAI_MODEL = "gpt-3.5-turbo"

furhat = FurhatRemoteAPI(ROBOT_IP)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a friendly movie expert. Use the provided movie information to answer the user's question accurately. If the information is insufficient, you can say you don't know, but try to be helpful based on the context. Keep answers concise and friendly."""

conversation_history = []
MAX_HISTORY = 5

async def get_gpt_response_with_rag(user_input):
    """Retrieve relevant movies and generate a response, using conversation context."""
    try:
        # Build a search query that includes the last mentioned movie/topic
        search_query = user_input
        if conversation_history:
            # Get the last assistant message (which might contain the movie title)
            last_assistant = None
            for msg in reversed(conversation_history):
                if msg["role"] == "assistant":
                    last_assistant = msg["content"]
                    break
            if last_assistant:
                # Extract potential movie titles (simple approach: take the first sentence)
                # Better: use the last user input that mentioned a movie? For now, just append.
                search_query = f"{user_input} {last_assistant[:100]}"
        
        print(f"🔍 Search query for retrieval: {search_query}")
        
        # Query knowledge base
        kb_results = movie_kb.query_movies(search_query, n_results=5)
        
        context = ""
        if kb_results and 'documents' in kb_results and kb_results['documents']:
            docs = kb_results['documents'][0]
            print("📚 Retrieved documents:")
            for i, doc in enumerate(docs):
                print(f"   {i+1}: {doc[:150]}...")
            context = "Relevant movie information:\n" + "\n---\n".join(docs)
        else:
            context = "No specific movie information found in the knowledge base."
        
        # Build messages with history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in conversation_history[-MAX_HISTORY*2:]:
            messages.append(msg)
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})
        
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=250
        )
        reply = response.choices[0].message.content
        
        # Update history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": reply})
        if len(conversation_history) > MAX_HISTORY * 2:
            conversation_history[:2] = []
        
        return reply
    except Exception as e:
        print(f"❌ Error in RAG response: {e}")
        return "Sorry, I'm having trouble accessing my movie knowledge right now."

async def conversation_loop():
    furhat.say(text="Hi! I'm your movie bot. Ask me anything about films!")
    
    consecutive_timeouts = 0
    
    while True:
        print("👂 Listening... (will wait up to 25 seconds)")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(furhat.listen),
                timeout=25.0
            )
            if result.success and result.message.strip():
                user_speech = result.message.strip()
                print(f"🗣️ User said: {user_speech}")
                consecutive_timeouts = 0
                
                bot_reply = await get_gpt_response_with_rag(user_speech)
                print(f"🤖 Bot replies: {bot_reply}")
                furhat.say(text=bot_reply)
            else:
                print("🔇 Silence or recognition issue, continuing to listen...")
                continue
                
        except asyncio.TimeoutError:
            consecutive_timeouts += 1
            print(f"⏰ Listening timed out ({consecutive_timeouts}).")
            if consecutive_timeouts == 1:
                furhat.say(text="Are you still there? Ask me about a movie!")
            else:
                print("Still no speech, waiting...")
            continue
        
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(conversation_loop())