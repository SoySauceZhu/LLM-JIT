"""Shared mortgage labels, versioned prompts and message construction."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("A", "B", "C", "D", "E")
PROMPT_DIR = ROOT / "code/prompts"
SUPPORTED_VERSIONS = ("v1", "v2", "v3", "v4")
DEFAULT_VERSION = "v3"


def load_prompt(version):
    if version == "v1":
        path = PROMPT_DIR / "mortgage_v1.txt"
        # Keep this snapshot compatible with the original Teacher ledger.
        return {"prompt_file": path.name, "system_prompt": path.read_text(encoding="utf-8").strip(),
                "user_template": "<complaint>\n{text}\n</complaint>"}
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unknown prompt version: {version}")
    path = PROMPT_DIR / f"mortgage_{version}.toml"
    with path.open("rb") as f:
        return {"prompt_file": path.name, **tomllib.load(f)}


DEFAULT_PROMPT = load_prompt(DEFAULT_VERSION)


def build_messages(text, prompt):
    render_user = prompt["user_template"].format
    messages = [{"role": "system", "content": prompt["system_prompt"]}]
    for example in prompt.get("examples", ()):
        messages.append({"role": "user", "content": render_user(text=example["text"])})
        messages.append({"role": "assistant", "content": example["label"]})
    final_user = render_user(text=text)
    if prompt.get("final_instruction"):
        final_user += "\n" + prompt["final_instruction"]
    messages.append({"role": "user", "content": final_user})
    return messages

def parse_label(raw):
    label = (raw or "").strip()
    return label if label in LABELS else None
