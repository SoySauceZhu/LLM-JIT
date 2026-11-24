"""Frozen CFPB cohort selection and shared Student-tokenizer clipping."""

import csv
import hashlib
import random
from collections import Counter

from mortgage_prompt import LABELS, ROOT

DATA = ROOT / "data/mortgage_complaint.csv"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
HEAD, TAIL = 2000, 1000
SPLIT_COUNTS = {"development": 957, "history": 8202, "validation": 1013, "test": 2024}
COHORT_HASH = "3c59aa321b65ea493b579cd503af0aeb580fa9c28ee4cabe4b15d136ab123dfa"


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_rows(rows, *, reference_snapshot=False):
    """Audit metadata and text identity without scoring any split."""
    ids, hashes = set(), set()
    previous_sequence = previous_time_id = None
    months = {"development": {1}, "history": set(range(2, 10)),
              "validation": {10}, "test": {11, 12}}
    for row in rows:
        cid, digest = row["complaint_id"], row["text_sha256"]
        sequence = int(row["sequence_no"])
        time_id = (row["received_at_utc"], int(cid))
        if cid in ids or digest in hashes:
            raise ValueError("Duplicate complaint ID or normalized text across the cohort")
        if not row["text"].strip() or sha256(row["text"]) != digest:
            raise ValueError(f"Empty or changed narrative: {cid}")
        if previous_sequence is not None and (sequence <= previous_sequence or time_id < previous_time_id):
            raise ValueError("Cohort is not in chronological sequence order")
        split = row["split"]
        date = row["received_at_utc"]
        if split not in months or not date.startswith("2024-") or int(date[5:7]) not in months[split]:
            raise ValueError(f"Invalid temporal split: {cid}")
        if row["reference_label"] not in LABELS:
            raise ValueError(f"Unknown reference label: {cid}")
        ids.add(cid)
        hashes.add(digest)
        previous_sequence, previous_time_id = sequence, time_id
    counts = dict(Counter(row["split"] for row in rows))
    fingerprint = sha256("".join(row["complaint_id"] + "\n" for row in rows))
    if reference_snapshot and (counts != SPLIT_COUNTS or fingerprint != COHORT_HASH):
        raise ValueError("Cohort differs from the frozen 2026-11-21 snapshot")
    return {"split_counts": counts, "ordered_ids_sha256": fingerprint}


def load_cohort(path=DATA):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    audit_rows(rows, reference_snapshot=True)
    return rows


def select_rows(rows, split, size=None, seed=42):
    selected = [row for row in rows if row["split"] == split]
    if split != "development":
        if size is not None:
            raise ValueError("--size is only supported for development")
        return selected
    size = 300 if size is None else size
    groups = {label: [r for r in selected if r["reference_label"] == label] for label in LABELS}
    total = len(selected)
    if not 1 <= size <= total:
        raise ValueError(f"--size must be between 1 and {total}")
    quotas = {label: size * len(group) for label, group in groups.items()}
    counts = {label: quota // total for label, quota in quotas.items()}
    by_remainder = sorted(LABELS, key=lambda label: -(quotas[label] % total))
    for label in by_remainder[:size - sum(counts.values())]:
        counts[label] += 1
    rng = random.Random(seed)
    return sorted([r for label in LABELS for r in rng.sample(groups[label], counts[label])],
                  key=lambda r: int(r["sequence_no"]))


def prepare_cases(rows, tokenizer=None):
    if tokenizer is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        path = hf_hub_download(TOKENIZER, "tokenizer.json", revision=REVISION)
        tokenizer = Tokenizer.from_file(path)
    cases = []
    for index, row in enumerate(rows, 1):
        text = row["text"]
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        clipped = len(ids) > HEAD + TAIL
        if clipped:
            text = tokenizer.decode(ids[:HEAD] + ids[-TAIL:], skip_special_tokens=False)
        cases.append({
            "complaint_id": row["complaint_id"], "reference_label": row["reference_label"],
            "sequence_no": int(row["sequence_no"]), "received_at_utc": row["received_at_utc"],
            "arrival_index": index, "text": text, "input_sha256": sha256(text),
            "student_tokens_before_clip": len(ids), "clipped": clipped,
        })
    return cases


def manifest_cases(cases):
    return [{**{k: v for k, v in case.items() if k != "text"},
             "input_sha256": sha256(case["text"])} for case in cases]


def default_output(backend, split, version, size=None, seed=42):
    suffix = f"_n{size or 300}_s{seed}" if split == "development" else ""
    return ROOT / "output" / f"baseline_{version}" / f"{backend}_{split}{suffix}"
