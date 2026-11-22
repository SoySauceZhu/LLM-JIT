"""Send one development complaint to the SiliconFlow Teacher.

Install dependencies with: python3 -m pip install openai python-dotenv
Set SILICONFLOW_API_KEY to a key from cloud.siliconflow.com in the environment
or the repository's .env file.
Run from the repository root with: python3 code/smoke_gemma.py
"""

import csv
from openai import APIError

from gemma_teacher import DATA, make_client, parse_reply, request_label


def main():
    client = make_client()
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        sample = next((r for r in rows if r["split"] == "development" and r["text"].strip()), None)
    if sample is None:
        raise SystemExit("No development complaint with a narrative found in the CSV.")

    try:
        response = request_label(client, sample["text"])
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
