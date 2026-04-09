import os
import asyncio
import datetime
from nemoguardrails import RailsConfig, LLMRails
import actions

LOG_FILE = "guardrails_test_log.txt"

def log_interaction(user_msg, bot_msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"{timestamp} | User: {user_msg}\n")
        f.write(f"{timestamp} | Bot: {bot_msg}\n\n")

async def main():
    # Load guardrails configuration
    config = RailsConfig.from_path("./guardrails_config")
    rails = LLMRails(config)
    # Register custom actions
    rails.register_action(actions.retrieve_movies, name="retrieve_movies")
    rails.register_action(actions.filter_by_director, name="filter_by_director")

    print("=== Guardrails Test Mode (Text Input) ===")
    print("Type your messages below. Type 'quit' to exit.\n")

    messages = []  # full conversation history

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        print("🚀 Calling guardrails generate_async...")
        response = await rails.generate_async(messages=messages)

        # Extract bot reply (handles different return types)
        if isinstance(response, dict) and "content" in response:
            bot_reply = response["content"]
        elif isinstance(response, str):
            bot_reply = response
        else:
            bot_reply = str(response)

        messages.append({"role": "assistant", "content": bot_reply})
        log_interaction(user_input, bot_reply)

        print(f"🤖 Bot: {bot_reply}\n")

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Guardrails Test Log\n")
        f.write("=" * 40 + "\n")
    asyncio.run(main())