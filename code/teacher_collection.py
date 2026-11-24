"""Run independent Teacher requests with bounded concurrency and a resumable ledger."""

import json
import sqlite3
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from functools import partial
from threading import Event, Lock

from openai import APIError, APITimeoutError

from teacher_client import TEACHER_MODELS, parse_reply, request_label
from run_store import single_runner


def collect_case(db, client, case, used, concurrency, stop, predict):
    """Return True for a terminal case. Leave interrupted work eligible for resume."""
    cid = case["complaint_id"]
    for attempt_number in range(used, 2):
        if stop.is_set():
            return False
        start = time.monotonic()
        raw = label = detail = response_id = response_model = usage = None
        try:
            response = predict(client, case["text"])
            raw, label = parse_reply(response)
            status = "ok" if label else "invalid"
            response_id, response_model, usage = response.id, response.model, response.usage
        except APITimeoutError as exc:
            status, detail = "timeout", str(exc)
        except APIError as exc:
            stop.set()
            status, detail = "paused", str(exc)
            if getattr(exc, "status_code", None) == 429:
                retry_after = exc.response.headers.get("retry-after")
                if retry_after:
                    detail += f" (Retry-After: {retry_after})"

        seconds = time.monotonic() - start
        with db:
            db.execute("""INSERT INTO attempts
                (complaint_id, status, raw_reply, label, detail, response_id, response_model,
                 prompt_tokens, completion_tokens, seconds, concurrency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cid, status, raw, label, detail, response_id, response_model,
                 usage.prompt_tokens if usage else None, usage.completion_tokens if usage else None,
                 seconds, concurrency))
        if status == "paused":
            raise SystemExit(f"Paused at {cid}: {detail}\nResume using the same output directory.")
        if status == "ok":
            break
    return True


def collect(db, db_path, client, cases, concurrency, *, predict=None):
    if predict is None:
        config = json.loads(db.execute("SELECT config FROM run WHERE id=1").fetchone()[0])
        teacher_model = next((name for name, profile in TEACHER_MODELS.items()
                              if profile["model"] == config["model"]), None)
        if teacher_model is None:
            raise ValueError(f"Unknown Teacher model in ledger: {config['model']}")
        predict = partial(request_label, prompt=config, teacher_model=teacher_model)
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    # Paused service errors do not consume the two classification attempts.
    history = db.execute("""SELECT complaint_id, COUNT(*), MAX(status = 'ok')
        FROM attempts WHERE status != 'paused' GROUP BY complaint_id""")
    used = {cid: 2 if accepted else count for cid, count, accepted in history}
    pending = deque((case, used.get(case["complaint_id"], 0)) for case in cases
                    if used.get(case["complaint_id"], 0) < 2)
    completed = len(cases) - len(pending)
    print(f"Already complete: {completed}/{len(cases)}. Pending: {len(pending)}. "
          f"Concurrency: {concurrency}.", flush=True)
    if not pending:
        return

    stop = Event()
    lock = Lock()

    def worker():
        nonlocal completed
        # Each worker owns its SQLite connection. Only inserts take a write lock.
        with closing(sqlite3.connect(db_path, timeout=60)) as writer:
            while not stop.is_set():
                with lock:
                    if not pending:
                        return
                    case, prior_attempts = pending.popleft()
                if collect_case(writer, client, case, prior_attempts, concurrency, stop, predict):
                    with lock:
                        completed += 1
                        if completed % 10 == 0 or completed == len(cases):
                            print(f"{completed}/{len(cases)}", flush=True)

    workers = min(concurrency, len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            futures = [pool.submit(worker) for _ in range(workers)]
            for future in as_completed(futures):
                future.result()
        finally:
            # On Ctrl+C or an error, workers finish and record requests already in flight.
            stop.set()
