import os
import asyncio
from furhat_remote_api import FurhatRemoteAPI
from nemoguardrails import RailsConfig, LLMRails
import actions

# Configuration
ROBOT_IP = "127.0.0.1"
furhat = FurhatRemoteAPI(ROBOT_IP)

# Load guardrails configuration
config = RailsConfig.from_path("./guardrails_config")
rails = LLMRails(config)

# Register custom actions
rails.register_action(actions.retrieve_movies, name="retrieve_movies")
rails.register_action(actions.filter_by_director, name="filter_by_director")

# Store full conversation history
conversation = []

async def conversation_loop():
    furhat.say("Hello! I'm your movie companion. Ask me about any film!")
    loop = asyncio.get_running_loop()

    while True:
        print("👂 Listening...")
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, furhat.listen),
                timeout=20.0
            )
            if result and result.success and result.message.strip():
                user_speech = result.message.strip()
                print(f"🗣️ User: {user_speech}")

                # Add user message to conversation history
                conversation.append({"role": "user", "content": user_speech})

                # Generate response with guardrails
                response = await rails.generate_async(messages=conversation)

                # Extract bot reply (handles different return types)
                if isinstance(response, dict) and "content" in response:
                    bot_reply = response["content"]
                elif isinstance(response, str):
                    bot_reply = response
                else:
                    bot_reply = str(response)

                # Add bot reply to history
                conversation.append({"role": "assistant", "content": bot_reply})

                print(f"🤖 Bot: {bot_reply}")
                furhat.say(bot_reply)
            else:
                print("No speech detected...")
        except asyncio.TimeoutError:
            print("Listening timeout.")
            furhat.say("Are you still there? Ask me about a movie!")

if __name__ == "__main__":
    asyncio.run(conversation_loop())