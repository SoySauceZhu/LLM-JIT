"""Preview, collect or export full mortgage Teacher baselines."""

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from functools import partial
from pathlib import Path

from evaluation import score
from mortgage_data import default_output, load_development, load_experiment, prepare_cases, select_rows
from mortgage_prompt import DEFAULT_PROMPT, DEFAULT_VERSION, SUPPORTED_VERSIONS, load_prompt
from run_store import (
    MILESTONES, attempts_by_case, final_attempt, initialize as initialize_store,
    make_config, progress_metrics, read_config, usage_metrics, write_exports,
)
from teacher_client import (
    BASE_URL, DECODING, MODEL, REQUEST_TIMEOUT_SECONDS, SDK_MAX_RETRIES, make_client, request_label,
)
from teacher_collection import collect, single_runner


def initialize(db, cases, seed, prompt=DEFAULT_PROMPT, *, split="development", milestones=MILESTONES):
    config = make_config(cases, split, seed, prompt, backend="teacher", model=MODEL,
                         decoding=DECODING, base_url=BASE_URL,
                         max_attempts=2, request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                         sdk_max_retries=SDK_MAX_RETRIES,
                         milestones=list(milestones) if split == "history" else [])
    initialize_store(db, config)
    return config


def export_results(db, cases, output):
    config = json.loads(db.execute("SELECT config FROM run WHERE id=1").fetchone()[0])
    grouped = attempts_by_case(db)
    fields = ["complaint_id", "reference_label", "teacher_label", "status", "attempts",
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
    summary = {**score(terminal, "teacher_label"), **progress_metrics(len(cases), len(terminal)),
               "clipped": sum(c["clipped"] for c in cases), "usage": usage_metrics(db),
               "attempts_by_concurrency": dict(db.execute(
                   "SELECT concurrency, COUNT(*) FROM attempts GROUP BY concurrency"))}
    json_files = {"config.json": config, "metrics.json": summary}
    csv_files = {"predictions.csv": (fields, rows)}
    if config["split"] == "history":
        cursor = db.execute("SELECT * FROM history_traces ORDER BY success_index")
        trace_fields = [c[0] for c in cursor.description]
        traces = [dict(zip(trace_fields, row)) for row in cursor]
        csv_files["history_labels.csv"] = (trace_fields, traces)
        prefixes = []
        frozen = {n: (arrival, members, digest) for n, arrival, members, digest in
                  db.execute("SELECT * FROM prefixes ORDER BY milestone")}
        for n in config["milestones"]:
            item = {"milestone": n, "reached": n in frozen}
            if n in frozen:
                arrival, members, digest = frozen[n]
                item.update(arrival_index=arrival, sha256=digest)
                json_files[f"prefixes/n{n}.json"] = {**item, "members": json.loads(members)}
            prefixes.append(item)
        summary["history_successes"] = len(traces)
        summary["history_cursor"] = db.execute("SELECT cursor FROM progress WHERE id=1").fetchone()[0]
        summary["milestones"] = prefixes
    write_exports(output, json_files=json_files, csv_files=csv_files)
    print(f"terminal={len(terminal)}/{len(cases)} complete={summary['complete']} "
          f"accuracy={summary['accuracy']} macro-F1={summary['macro_f1']} output={output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "history", "validation"), default="validation")
    parser.add_argument("--size", type=int, help="Development sample only (default: 300)")
    parser.add_argument("--seed", type=int, help="Development sampling seed only (default: 42)")
    parser.add_argument("--prompt-version", choices=SUPPORTED_VERSIONS, default=DEFAULT_VERSION)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.split != "development" and (args.size is not None or args.seed is not None):
        parser.error("--size and --seed are only supported for development")
    if args.concurrency < 1 or (args.split == "history" and args.concurrency != 1):
        parser.error("Concurrency must be positive; history requires --concurrency 1")
    seed = args.seed if args.seed is not None else 42
    output = args.output or default_output("teacher", args.split, args.prompt_version, args.size, seed)
    if args.export_only:
        config = read_config(output)
        if config.get("schema_version") != 2 or config.get("backend") != "teacher":
            parser.error("Export requires a new Teacher baseline ledger; archived pilots stay read-only")
        with single_runner(output), closing(sqlite3.connect(output / "run.sqlite")) as db:
            export_results(db, config["cases"], output)
        return
    load_data = load_development if args.split == "development" else load_experiment
    rows = select_rows(load_data(), args.split, args.size, seed)
    print(f"{args.split} n={len(rows)} counts={dict(Counter(r['reference_label'] for r in rows))} "
          f"prompt={args.prompt_version} output={output}")
    if not args.run:
        print("Preview only. Add --run to start.")
        return
    cases = prepare_cases(rows)
    prompt = load_prompt(args.prompt_version)
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "run.sqlite"
    with single_runner(output), closing(sqlite3.connect(db_path, timeout=60)) as db:
        initialize(db, cases, seed, prompt, split=args.split)
        try:
            # Credentials and network access are needed only for pending requests.
            grouped = attempts_by_case(db)
            if any(not final_attempt(grouped.get(c["complaint_id"], [])) for c in cases):
                with make_client() as client:
                    collect(db, db_path, client, cases, args.concurrency,
                            predict=partial(request_label, prompt=prompt))
        finally:
            export_results(db, cases, output)


if __name__ == "__main__":
    main()
