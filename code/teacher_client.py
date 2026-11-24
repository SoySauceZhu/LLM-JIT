"""SiliconFlow Gemma Teacher client."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from mortgage_prompt import DEFAULT_PROMPT, ROOT, build_messages, parse_label

MODEL = "google/gemma-4-31B-it"
BASE_URL = "https://api.siliconflow.com/v1"
DECODING = {"temperature": 0, "max_tokens": 16}
REQUEST_TIMEOUT_SECONDS = 45
SDK_MAX_RETRIES = 0


def make_client():
    load_dotenv(ROOT / ".env")
    key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set SILICONFLOW_API_KEY in .env or the environment.")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS,
                  max_retries=SDK_MAX_RETRIES)


def request_label(client, text, prompt=DEFAULT_PROMPT):
    return client.chat.completions.create(
        model=MODEL,
        messages=build_messages(text, prompt),
        **DECODING,
    )


def parse_reply(response):
    raw = response.choices[0].message.content if response.choices else None
    return raw, parse_label(raw)
