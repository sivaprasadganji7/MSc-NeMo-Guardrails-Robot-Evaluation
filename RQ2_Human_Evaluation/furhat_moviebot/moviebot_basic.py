import os
import asyncio
from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI

# Configuration
ROBOT_IP = "127.0.0.1"   # <-- CHANGE to your robot's IP
OPENAI_MODEL = "gpt-3.5-turbo"   # or "gpt-4"

# Initialize clients
furhat = FurhatRemoteAPI(ROBOT_IP)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt to keep the bot focused on movies
SYSTEM_PROMPT = "You are a friendly movie expert. Answer questions about movies, actors, directors, and recommendations. If the user asks something completely off-topic, politely steer the conversation back to movies."

async def get_gpt_response(user_input):
    """Call OpenAI API and return the assistant's message."""
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I'm having trouble thinking right now."

async def conversation_loop():
    furhat.say(text="Hi! I'm your movie bot. Ask me anything about films!")
    
    loop = asyncio.get_running_loop()
    
    while True:
        print("Listening... (timeout 10 seconds)")
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, furhat.listen),
                timeout=10.0
            )
            if result and result.success and result.message.strip():
                user_speech = result.message.strip()
                print(f"User said: {user_speech}")
            else:
                print("No valid speech detected.")
                furhat.say(text="Sorry, I didn't catch that. Ask me about a movie!")
                continue
        except asyncio.TimeoutError:
            print("Listening timed out.")
            furhat.say(text="Are you still there? Ask me about a movie!")
            continue
        
        bot_reply = await get_gpt_response(user_speech)
        print(f"Bot replies: {bot_reply}")
        
        furhat.say(text=bot_reply)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(conversation_loop())