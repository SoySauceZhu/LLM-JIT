"""Preview, collect or export full mortgage Teacher baselines."""

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from functools import partial
from pathlib import Path

from evaluation import score
from mortgage_data import COHORT_SPLIT_COUNTS, load_experiment, prepare_cases, select_rows
from mortgage_prompt import DEFAULT_PROMPT, DEFAULT_VERSION, SUPPORTED_VERSIONS, load_prompt
from run_store import (
    attempts_by_case, final_attempt, initialize as initialize_store,
    make_config, progress_metrics, read_config, usage_metrics, write_exports,
)
from teacher_client import (
    BASE_URL, DECODING, DEFAULT_TEACHER_MODEL, TEACHER_MODELS,
    REQUEST_TIMEOUT_SECONDS, SDK_MAX_RETRIES, make_client, request_label, teacher_output,
)
from teacher_collection import collect, single_runner


def initialize(db, cases, prompt=DEFAULT_PROMPT, *, split=None, teacher_model=DEFAULT_TEACHER_MODEL):
    profile = TEACHER_MODELS[teacher_model]
    # Keep the optional Gemma config compatible with existing schema-3 runs.
    extra = {"request_extra_body": dict(profile["extra_body"])} if profile["extra_body"] else {}
    config = make_config(cases, split, prompt, backend="teacher", model=profile["model"],
                         decoding=DECODING, base_url=BASE_URL,
                         max_attempts=2, request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                         sdk_max_retries=SDK_MAX_RETRIES, **extra)
    initialize_store(db, config)
    return config


def export_results(db, cases, output):
    config = json.loads(db.execute("SELECT config FROM run WHERE id=1").fetchone()[0])
    grouped = attempts_by_case(db)
    fields = ["complaint_id", "split", "reference_label", "teacher_label", "status", "attempts",
              "sequence_no", "received_at_utc", "arrival_index", "input_sha256",
              "student_tokens_before_clip", "clipped"]
    rows, terminal = [], []
    for case in cases:
        attempts = grouped.get(case["complaint_id"], [])
        final = final_attempt(attempts)
        row = {field: case.get(field) for field in fields}
        row.update(teacher_label=final["label"] if final else None,
                   status=final["status"] if final else "pending", attempts=len(attempts))
        rows.append(row)
        if final:
            terminal.append(row)
    summary = {"model": config["model"], "base_url": config.get("base_url"),
               **score(terminal, "teacher_label"), **progress_metrics(len(cases), len(terminal)),
               "clipped": sum(c["clipped"] for c in cases), "usage": usage_metrics(db),
               "attempts_by_concurrency": dict(db.execute(
                   "SELECT concurrency, COUNT(*) FROM attempts GROUP BY concurrency"))}
    json_files = {"config.json": config, "metrics.json": summary}
    csv_files = {"predictions.csv": (fields, rows)}
    write_exports(output, json_files=json_files, csv_files=csv_files)
    print(f"terminal={len(terminal)}/{len(cases)} complete={summary['complete']} "
          f"accuracy={summary['accuracy']} macro-F1={summary['macro_f1']} output={output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", choices=TEACHER_MODELS, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--split", choices=COHORT_SPLIT_COUNTS,
                        help="Optional partition filter; omit to label all 12,196 complaints")
    parser.add_argument("--prompt-version", choices=SUPPORTED_VERSIONS, default=DEFAULT_VERSION)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("Concurrency must be positive")
    output = args.output or teacher_output(args.teacher_model, args.split, args.prompt_version)
    if args.export_only:
        config = read_config(output)
        if config.get("schema_version") not in (2, 3) or config.get("backend") != "teacher":
            parser.error("Export requires a schema 2 or 3 Teacher baseline ledger")
        with single_runner(output), closing(sqlite3.connect(output / "run.sqlite")) as db:
            export_results(db, config["cases"], output)
        return
    rows = select_rows(load_experiment(), args.split)
    print(f"{args.split or 'all'} n={len(rows)} splits={dict(Counter(r['split'] for r in rows))} "
          f"counts={dict(Counter(r['reference_label'] for r in rows))} "
          f"model={args.teacher_model} api_model={TEACHER_MODELS[args.teacher_model]['model']} "
          f"prompt={args.prompt_version} output={output}")
    if not args.run:
        print("Preview only. Add --run to start.")
        return
    cases = prepare_cases(rows)
    prompt = load_prompt(args.prompt_version)
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "run.sqlite"
    with single_runner(output), closing(sqlite3.connect(db_path, timeout=60)) as db:
        initialize(db, cases, prompt, split=args.split, teacher_model=args.teacher_model)
        try:
            # Credentials and network access are needed only for pending requests.
            grouped = attempts_by_case(db)
            if any(not final_attempt(grouped.get(c["complaint_id"], [])) for c in cases):
                with make_client() as client:
                    collect(db, db_path, client, cases, args.concurrency,
                            predict=partial(request_label, prompt=prompt, teacher_model=args.teacher_model))
        finally:
            export_results(db, cases, output)


if __name__ == "__main__":
    main()
