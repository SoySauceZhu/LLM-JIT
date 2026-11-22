"""Shared SiliconFlow client and mortgage classification prompt."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mortgage_complaint.csv"
MODEL = "google/gemma-4-31B-it"
BASE_URL = "https://api.siliconflow.com/v1"
LABELS = ("A", "B", "C", "D", "E")
PROMPT_PATH = ROOT / "code/prompts/mortgage_v1.txt"
PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
USER_TEMPLATE = "<complaint>\n{text}\n</complaint>"
DECODING = {"temperature": 0, "max_tokens": 16}


def make_client():
    load_dotenv(ROOT / ".env")
    key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set SILICONFLOW_API_KEY in .env or the environment.")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=45, max_retries=0)


def request_label(client, text):
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ],
        **DECODING,
    )


def parse_reply(response):
    raw = response.choices[0].message.content if response.choices else None
    label = (raw or "").strip()
    return raw, label if label in LABELS else None
