# llm.py
from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI
from nemoguardrails import LLMRails, RailsConfig


# ========================
# BASELINE LLM CALL
# ========================
def chat_completion(system: str, user: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )

    return resp.choices[0].message.content.strip()


# ========================
# HARDENED (NeMo Guardrails)
# ========================
_rails = None


def hardened_chat_completion(user: str) -> tuple[str, bool]:
    global _rails

    if _rails is None:
        base_dir = Path(__file__).resolve().parent
        config_dir = base_dir / "config"

        if not config_dir.exists():
            raise RuntimeError(
                f"NeMo Guardrails config directory not found at: {config_dir}"
            )

        config = RailsConfig.from_path(str(config_dir))
        _rails = LLMRails(config)

    response = _rails.generate(user).strip()

    # Simple, explicit signal
    blocked = response.startswith("[BLOCKED]")

    # Clean before sending to Furhat
    response = response.replace("[BLOCKED] ", "").replace("[MODIFIED] ", "")

    return response, blocked
