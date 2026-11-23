"""Shared SiliconFlow client and mortgage classification prompt."""

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mortgage_complaint.csv"
MODEL = "google/gemma-4-31B-it"
BASE_URL = "https://api.siliconflow.com/v1"
LABELS = ("A", "B", "C", "D", "E")
PROMPT_DIR = ROOT / "code/prompts"
DEFAULT_VERSION = "v2"
DECODING = {"temperature": 0, "max_tokens": 16}


def load_prompt(version):
    if version == "v1":
        path = PROMPT_DIR / "mortgage_v1.txt"
        # Keep this snapshot compatible with the original Teacher ledger.
        return {"prompt_file": path.name, "system_prompt": path.read_text(encoding="utf-8").strip(),
                "user_template": "<complaint>\n{text}\n</complaint>"}
    if version != "v2":
        raise ValueError(f"Unknown prompt version: {version}")
    path = PROMPT_DIR / "mortgage_v2.toml"
    with path.open("rb") as f:
        return {"prompt_file": path.name, **tomllib.load(f)}


DEFAULT_PROMPT = load_prompt(DEFAULT_VERSION)


def build_messages(text, prompt):
    render_user = prompt["user_template"].format
    messages = [{"role": "system", "content": prompt["system_prompt"]}]
    for example in prompt.get("examples", ()):
        messages.append({"role": "user", "content": render_user(text=example["text"])})
        messages.append({"role": "assistant", "content": example["label"]})
    messages.append({"role": "user", "content": render_user(text=text)})
    return messages


def make_client():
    load_dotenv(ROOT / ".env")
    key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set SILICONFLOW_API_KEY in .env or the environment.")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=45, max_retries=0)


def request_label(client, text, prompt=DEFAULT_PROMPT):
    return client.chat.completions.create(
        model=MODEL,
        messages=build_messages(text, prompt),
        **DECODING,
    )


def parse_reply(response):
    raw = response.choices[0].message.content if response.choices else None
    label = (raw or "").strip()
    return raw, label if label in LABELS else None
