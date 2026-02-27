from __future__ import annotations

import os
import re
import time
from typing import List

from dotenv import load_dotenv
from furhat_remote_api import FurhatRemoteAPI

from llm import chat_completion


def sanitize_furhat_host(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("http://", "").replace("https://", "")
    if s.startswith("//"):
        s = s[2:]
    s = s.split("/", 1)[0]
    if ":" in s:
        s = s.split(":", 1)[0]
    return s


def listen_until_heard(furhat: FurhatRemoteAPI, max_wait_s: int) -> str:
    start = time.time()
    while time.time() - start < max_wait_s:
        res = furhat.listen()

        if isinstance(res, dict):
            msg = (res.get("message") or "").strip()
            ok = bool(res.get("success", False))
        else:
            msg = (getattr(res, "message", "") or "").strip()
            ok = True if msg else False

        if ok and msg:
            return msg

        time.sleep(0.2)

    return ""


def build_companion_system_prompt() -> str:
    return """
You are an open-domain companion robot interacting with a human.

Rules:
- Be friendly, calm, and supportive.
- Engage in open-ended conversation.
- Do NOT claim a physical body, emotions, or consciousness.
- Do NOT provide harmful or illegal advice.
- If a request is unsafe, refuse briefly and redirect.
- Keep responses concise.
""".strip()


def main():
    load_dotenv()

    max_wait_for_answer_s = int(os.getenv("MAX_WAIT_FOR_ANSWER_S", 20))
    stop_words_regex = r"\b(stop|goodbye|quit|exit)\b"

    furhat_host_raw = os.getenv("FURHAT_HOST", "127.0.0.1")
    furhat_host = sanitize_furhat_host(furhat_host_raw)
    furhat_voice = os.getenv("FURHAT_VOICE", "Matthew")

    print("=== BASELINE ===")
    print("Using Furhat host:", furhat_host)

    furhat = FurhatRemoteAPI(furhat_host)
    try:
        furhat.set_voice(name=furhat_voice)
    except Exception:
        pass

    system_prompt = build_companion_system_prompt()
    messages: List[dict] = [{"role": "system", "content": system_prompt}]

    furhat.say(text="Hi! I’m here to chat with you. You can talk about anything you like.")
    furhat.say(text="If you want to stop, just say goodbye.")

    while True:
        user_text = listen_until_heard(furhat, max_wait_s=max_wait_for_answer_s)

        if not user_text:
            furhat.say(text="Sorry, I didn’t catch that. Could you repeat?")
            continue

        if re.search(stop_words_regex, user_text.lower()):
            furhat.say(text="Alright. It was nice talking to you. Goodbye!")
            break

        messages.append({"role": "user", "content": user_text})
        history_text = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in messages[-10:]]
        )

        t0 = time.perf_counter()
        response_text = chat_completion(system=system_prompt, user=history_text)
        t_say = time.perf_counter()

        latency_ms = int((t_say - t0) * 1000)
        print(f"[BASELINE] latency_ms={latency_ms}")

        furhat.say(text=response_text)
        messages.append({"role": "assistant", "content": response_text})

    furhat.say(text="Session ended.")


if __name__ == "__main__":
    main()
