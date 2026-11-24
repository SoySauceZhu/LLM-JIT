"""Selectable SiliconFlow Teacher clients and frozen request profiles."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from mortgage_data import default_output
from mortgage_prompt import DEFAULT_PROMPT, ROOT, build_messages, parse_label

TEACHER_MODELS = {
    "deepseek-v4.1-flash-1017": {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "extra_body": {"enable_thinking": False},
    },
    "gemma4-31b-it": {"model": "google/gemma-4-31B-it", "extra_body": {}},
}
DEFAULT_TEACHER_MODEL = "deepseek-v4.1-flash-1017"
BASE_URL = "https://api.siliconflow.com/v1"
DECODING = {"temperature": 0, "max_tokens": 16}
REQUEST_TIMEOUT_SECONDS = 45
SDK_MAX_RETRIES = 0


def teacher_output(teacher_model, split, version):
    """Keep every Teacher baseline isolated, including optional partition filters."""
    if teacher_model not in TEACHER_MODELS:
        raise ValueError(f"Unknown Teacher profile: {teacher_model}")
    return default_output(f"teacher_{teacher_model}", split, version)


def make_client():
    load_dotenv(ROOT / ".env")
    key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set SILICONFLOW_API_KEY in .env or the environment.")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS,
                  max_retries=SDK_MAX_RETRIES)


def request_label(client, text, prompt=DEFAULT_PROMPT, *, teacher_model=DEFAULT_TEACHER_MODEL):
    profile = TEACHER_MODELS[teacher_model]
    options = {"extra_body": dict(profile["extra_body"])} if profile["extra_body"] else {}
    return client.chat.completions.create(
        model=profile["model"],
        messages=build_messages(text, prompt),
        **DECODING,
        **options,
    )


def parse_reply(response):
    raw = response.choices[0].message.content if response.choices else None
    return raw, parse_label(raw)
