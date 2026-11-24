"""Send one development complaint to the SiliconFlow Teacher.

Install dependencies with: uv sync --locked
Set SILICONFLOW_API_KEY to a key from cloud.siliconflow.com in the environment
or the repository's .env file.
Run from the repository root with: uv run --locked python code/teacher_smoke.py
"""

import argparse
from openai import APIError

from mortgage_data import load_development, prepare_cases
from mortgage_prompt import DEFAULT_VERSION, SUPPORTED_VERSIONS, load_prompt
from teacher_client import make_client, parse_reply, request_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-version", choices=SUPPORTED_VERSIONS, default=DEFAULT_VERSION)
    args = parser.parse_args()
    sample = load_development()[0]
    case = prepare_cases([sample])[0]

    try:
        with make_client() as client:
            response = request_label(client, case["text"], load_prompt(args.prompt_version))
    except APIError as exc:
        raise SystemExit(f"SiliconFlow request failed: {exc}") from None

    raw, label = parse_reply(response)
    if label is None:
        raise SystemExit(f"Expected a single label A–E, received {raw!r}.")

    print(f"PASS model={response.model} complaint_id={sample['complaint_id']} label={label}")
    if response.usage:
        print(f"prompt_tokens={response.usage.prompt_tokens} completion_tokens={response.usage.completion_tokens}")


if __name__ == "__main__":
    main()
