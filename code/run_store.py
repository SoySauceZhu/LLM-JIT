"""Shared baseline ledgers and atomic exports; archived pilot ledgers stay read-only."""

import csv
import hashlib
import json
import platform
import sqlite3
from contextlib import closing, contextmanager
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from mortgage_data import COHORT_HASH, DATA, HEAD, REVISION, SPLIT_COUNTS, TAIL, TOKENIZER, manifest_cases, sha256

MILESTONES = (100, 200, 400, 800, 1600, 3200, 6400)


@contextmanager
def single_runner(output):
    # The separate lock leaves the attempt ledger readable during inference.
    with closing(sqlite3.connect(output / "runner.lock.sqlite", timeout=0)) as lock:
        try:
            lock.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                raise SystemExit(f"Another baseline runner is using {output}.") from None
            raise
        yield


def read_config(run):
    uri = (Path(run).resolve() / "run.sqlite").as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        return json.loads(db.execute("SELECT config FROM run WHERE id=1").fetchone()[0])


def make_config(cases, split, seed, prompt, *, backend, model, decoding, **extra):
    packages = ("openai", "python-dotenv", "huggingface-hub", "tokenizers", "httpx")
    if backend == "student":
        packages += ("torch", "transformers")
    return {
        "schema_version": 2, "backend": backend, "model": model, "decoding": decoding,
        **prompt, "tokenizer": TOKENIZER, "revision": REVISION, "head": HEAD, "tail": TAIL,
        "split": split, "seed": seed if split == "development" else None,
        "cases": manifest_cases(cases), "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "data_path": "data/mortgage_complaint.csv", "cohort_ordered_ids_sha256": COHORT_HASH,
        "cohort_split_counts": SPLIT_COUNTS,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{p: version(p) for p in packages}}, **extra,
    }


def initialize(db, config):
    serialized = json.dumps(config, sort_keys=True)
    with db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS run (id INTEGER PRIMARY KEY, config TEXT NOT NULL)")
        existing = db.execute("SELECT config FROM run WHERE id=1").fetchone()
        if existing and existing[0] != serialized:
            raise ValueError("Inputs, prompt or configuration changed, or this is an archived pilot. Use a new --output directory.")
        db.execute("INSERT OR IGNORE INTO run VALUES (1, ?)", (serialized,))
        db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, complaint_id TEXT NOT NULL, status TEXT NOT NULL,
            raw_reply TEXT, label TEXT, detail TEXT, response_id TEXT, response_model TEXT,
            prompt_tokens INTEGER, completion_tokens INTEGER, seconds REAL NOT NULL,
            recorded_at TEXT DEFAULT CURRENT_TIMESTAMP, concurrency INTEGER NOT NULL DEFAULT 1,
            first_token_choice TEXT)""")
        db.execute("CREATE INDEX IF NOT EXISTS by_complaint ON attempts(complaint_id)")
        db.execute("""CREATE TABLE IF NOT EXISTS history_traces (
            success_index INTEGER PRIMARY KEY, arrival_index INTEGER NOT NULL UNIQUE,
            complaint_id TEXT NOT NULL UNIQUE, attempt_id INTEGER NOT NULL UNIQUE,
            teacher_label TEXT NOT NULL, input_sha256 TEXT NOT NULL,
            sequence_no INTEGER NOT NULL, received_at_utc TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS prefixes (
            milestone INTEGER PRIMARY KEY, arrival_index INTEGER NOT NULL,
            members TEXT NOT NULL, sha256 TEXT NOT NULL)""")
        db.execute("CREATE TABLE IF NOT EXISTS progress (id INTEGER PRIMARY KEY, cursor INTEGER NOT NULL)")
        db.execute("INSERT OR IGNORE INTO progress VALUES (1, 0)")


def record_history(db, case, attempt_id, label, milestones):
    """Called in the final attempt transaction, only by the serial history collector."""
    arrival = case["arrival_index"]
    cursor = db.execute("SELECT cursor FROM progress WHERE id=1").fetchone()[0]
    if arrival != cursor + 1:
        raise ValueError("History must commit in frozen arrival order")
    if label:
        success = db.execute("SELECT COUNT(*) FROM history_traces").fetchone()[0] + 1
        db.execute("INSERT INTO history_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            success, arrival, case["complaint_id"], attempt_id, label, case["input_sha256"],
            case["sequence_no"], case["received_at_utc"],
        ))
        if success in milestones:
            cursor_rows = db.execute("SELECT * FROM history_traces ORDER BY success_index")
            members = [dict(zip([c[0] for c in cursor_rows.description], row)) for row in cursor_rows]
            serialized = json.dumps(members, sort_keys=True)
            db.execute("INSERT INTO prefixes VALUES (?, ?, ?, ?)",
                       (success, arrival, serialized, sha256(serialized)))
    db.execute("UPDATE progress SET cursor=? WHERE id=1", (arrival,))


def attempts_by_case(db):
    cursor = db.execute("SELECT * FROM attempts ORDER BY id")
    columns = [c[0] for c in cursor.description]
    grouped = {}
    for row in cursor:
        attempt = dict(zip(columns, row))
        grouped.setdefault(attempt["complaint_id"], []).append(attempt)
    return grouped


def final_attempt(attempts):
    classification = [a for a in attempts if a["status"] != "paused"]
    accepted = next((a for a in classification if a["status"] == "ok"), None)
    return accepted or (classification[-1] if len(classification) >= 2 else None)


def progress_metrics(expected, terminal):
    return {"n_expected": expected, "n_terminal": terminal, "n_pending": expected - terminal,
            "complete": terminal == expected, "provisional": terminal != expected,
            "metric_denominator": "terminal_cases"}


def usage_metrics(db):
    n, prompt, completion, missing, seconds = db.execute("""SELECT COUNT(*), SUM(prompt_tokens),
        SUM(completion_tokens), SUM(prompt_tokens IS NULL OR completion_tokens IS NULL),
        SUM(seconds) FROM attempts""").fetchone()
    return {"attempts": n, "known_prompt_tokens": prompt, "known_completion_tokens": completion,
            "attempts_missing_usage": missing or 0, "attempt_seconds": seconds or 0,
            "cost_usd": None, "pricing_status": "unpriced"}


def write_exports(output, *, json_files, csv_files):
    """Stage all files; each destination replacement is atomic."""
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        for name, value in json_files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        for name, (fields, rows) in csv_files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        for name in (*json_files, *csv_files):
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / name).replace(destination)
