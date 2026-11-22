"""Preview or run a stratified January pilot for the mortgage Teacher."""

import argparse
import csv
import hashlib
import json
import random
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from gemma_teacher import (
    BASE_URL, DATA, DECODING, LABELS, MODEL, PROMPT, PROMPT_PATH, ROOT,
    USER_TEMPLATE, make_client,
)
from teacher_collection import collect, single_runner


TOKENIZER = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
HEAD, TAIL = 2000, 1000


def sample_development(size, seed):
    groups = {label: [] for label in LABELS}
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "development":
                groups[row["reference_label"]].append(row)
    total = sum(map(len, groups.values()))
    if not 1 <= size <= total:
        raise SystemExit(f"--size must be between 1 and {total}.")

    # Largest-remainder allocation preserves class proportions at integer sample sizes.
    quotas = {label: size * len(group) for label, group in groups.items()}
    counts = {label: quota // total for label, quota in quotas.items()}
    by_remainder = sorted(LABELS, key=lambda label: -(quotas[label] % total))
    for label in by_remainder[:size - sum(counts.values())]:
        counts[label] += 1
    rng = random.Random(seed)
    rows = [r for label in LABELS for r in rng.sample(groups[label], counts[label])]
    return sorted(rows, key=lambda r: int(r["sequence_no"]))


def prepare_cases(rows):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(TOKENIZER, "tokenizer.json", revision=REVISION)
    tokenizer = Tokenizer.from_file(path)
    cases = []
    for row in rows:
        text = row["text"]
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        clipped = len(ids) > HEAD + TAIL
        if clipped:
            text = tokenizer.decode(ids[:HEAD] + ids[-TAIL:], skip_special_tokens=False)
        cases.append({
            "complaint_id": row["complaint_id"], "reference_label": row["reference_label"],
            "text": text, "student_tokens_before_clip": len(ids), "clipped": clipped,
        })
    return cases


def initialize(db, cases, seed):
    # Freeze the exact visible inputs without storing a second copy of each narrative.
    manifest = [{
        **{k: v for k, v in case.items() if k != "text"},
        "input_sha256": hashlib.sha256(case["text"].encode()).hexdigest(),
    } for case in cases]
    config = json.dumps({
        "model": MODEL, "base_url": BASE_URL, "decoding": DECODING,
        "prompt_file": PROMPT_PATH.name, "system_prompt": PROMPT, "user_template": USER_TEMPLATE,
        "tokenizer": TOKENIZER, "revision": REVISION, "head": HEAD, "tail": TAIL,
        "split": "development", "seed": seed, "cases": manifest,
    }, sort_keys=True)
    with db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS run (id INTEGER PRIMARY KEY, config TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO run VALUES (1, ?)", (config,))
        if db.execute("SELECT config FROM run WHERE id=1").fetchone()[0] != config:
            raise SystemExit("Sample, prompt or configuration changed. Use a new --output directory.")
        db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, complaint_id TEXT NOT NULL, status TEXT NOT NULL,
            raw_reply TEXT, label TEXT, detail TEXT, response_id TEXT, response_model TEXT,
            prompt_tokens INTEGER, completion_tokens INTEGER, seconds REAL NOT NULL,
            recorded_at TEXT DEFAULT CURRENT_TIMESTAMP, concurrency INTEGER NOT NULL DEFAULT 1)""")
        db.execute("CREATE INDEX IF NOT EXISTS by_complaint ON attempts(complaint_id)")
        columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
        if "concurrency" not in columns:
            # All attempts from earlier versions ran serially.
            db.execute("ALTER TABLE attempts ADD COLUMN concurrency INTEGER NOT NULL DEFAULT 1")


def export_results(db, cases, output):
    matrix = {ref: dict.fromkeys((*LABELS, "FAIL"), 0) for ref in LABELS}
    clipped = 0
    fields = ["complaint_id", "reference_label", "teacher_label", "status", "attempts",
              "student_tokens_before_clip", "clipped"]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            cid = case["complaint_id"]
            attempts = db.execute(
                "SELECT status, label FROM attempts WHERE complaint_id=? ORDER BY id", (cid,),
            ).fetchall()
            status, label = attempts[-1]
            matrix[case["reference_label"]][label or "FAIL"] += 1
            clipped += case["clipped"]
            writer.writerow({
                **{k: v for k, v in case.items() if k != "text"},
                "teacher_label": label, "status": status, "attempts": len(attempts),
            })

    n = len(cases)
    per_class = {}
    for label in LABELS:
        support = sum(matrix[label].values())
        predicted = sum(row[label] for row in matrix.values())
        tp = matrix[label][label]
        per_class[label] = {
            "support": support, "recall": tp / support if support else None,
            "fpr": (predicted - tp) / (n - support) if support < n else None,
            "f1": 2 * tp / (support + predicted) if support + predicted else 0.0,
        }
    usage = db.execute("""SELECT COUNT(*) AS attempts, SUM(prompt_tokens) AS known_prompt_tokens,
        SUM(completion_tokens) AS known_completion_tokens,
        SUM(prompt_tokens IS NULL OR completion_tokens IS NULL) AS attempts_missing_usage
        FROM attempts""")
    summary = {
        "n": n, "clipped": clipped,
        "valid_rate": 1 - sum(row["FAIL"] for row in matrix.values()) / n,
        "accuracy": sum(matrix[label][label] for label in LABELS) / n,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(LABELS),
        "per_class": per_class, "confusion_rows_reference_columns_teacher": matrix,
        "usage": dict(zip([col[0] for col in usage.description], usage.fetchone())),
        "attempts_by_concurrency": dict(db.execute(
            "SELECT concurrency, COUNT(*) FROM attempts GROUP BY concurrency")),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"accuracy={summary['accuracy']:.3f}  macro-F1={summary['macro_f1']:.3f}  "
          f"valid={summary['valid_rate']:.1%}  output={output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1, metavar="N",
                        help="Maximum number of in-flight Teacher requests (default: 1)")
    parser.add_argument("--output", type=Path, help="Run directory, also used for resume")
    parser.add_argument("--run", action="store_true", help="Call the Teacher and score the sample")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    rows = sample_development(args.size, args.seed)
    counts = Counter(row["reference_label"] for row in rows)
    print(f"development n={len(rows)} seed={args.seed}  " +
          "  ".join(f"{label}={counts[label]}" for label in LABELS))
    if not args.run:
        print(f"Preview only. Prompt: {PROMPT_PATH}. Add --run to start.")
        return

    output = args.output or ROOT / "output" / f"gemma_dev_n{args.size}_s{args.seed}_v1"
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "run.sqlite"
    with single_runner(output), make_client() as client, closing(sqlite3.connect(db_path, timeout=60)) as db:
        cases = prepare_cases(rows)
        initialize(db, cases, args.seed)
        collect(db, db_path, client, cases, args.concurrency)
        export_results(db, cases, output)


if __name__ == "__main__":
    main()
