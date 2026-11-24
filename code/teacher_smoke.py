"""Send the first history complaint to the SiliconFlow Teacher.

Install dependencies with: uv sync --locked
Set SILICONFLOW_API_KEY to a key from cloud.siliconflow.com in the environment
or the repository's .env file.
Run from the repository root with: uv run --locked python code/teacher_smoke.py
"""

import argparse
from openai import APIError

from mortgage_data import load_experiment, prepare_cases, select_rows
from mortgage_prompt import DEFAULT_VERSION, SUPPORTED_VERSIONS, load_prompt
from teacher_client import DEFAULT_TEACHER_MODEL, TEACHER_MODELS, make_client, parse_reply, request_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", choices=TEACHER_MODELS, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--prompt-version", choices=SUPPORTED_VERSIONS, default=DEFAULT_VERSION)
    args = parser.parse_args()
    sample = select_rows(load_experiment(), "history")[0]
    case = prepare_cases([sample])[0]

    try:
        with make_client() as client:
            response = request_label(client, case["text"], load_prompt(args.prompt_version),
                                     teacher_model=args.teacher_model)
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
