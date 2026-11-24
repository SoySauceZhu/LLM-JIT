"""Evaluate the original SmolLM2 on the exact inputs of the 300-case Gemma pilot."""

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from baseline_gemma import REVISION, TOKENIZER, classification_scores, prepare_cases
from gemma_teacher import (
    DATA, DEFAULT_VERSION, LABELS, ROOT, SUPPORTED_VERSIONS, build_messages, load_prompt,
)


PILOT = ROOT / "output/gemma_dev_n300_s42_v1"
RESULTS = ROOT / "result/pre-run 300 sample test"


def read_config(run):
    uri = (run / "run.sqlite").as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        return json.loads(db.execute("SELECT config FROM run WHERE id=1").fetchone()[0])


def load_paired_cases(version=DEFAULT_VERSION):
    config = read_config(PILOT)
    manifest = config["cases"]
    ids = [entry["complaint_id"] for entry in manifest]
    selected = set(ids)
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        rows = {r["complaint_id"]: r for r in csv.DictReader(f)
                if r["split"] == "development" and r["complaint_id"] in selected}
    if len(selected) != 300 or len(ids) != 300:
        raise ValueError("Expected 300 unique complaints in the frozen pilot.")
    prompt = load_prompt(version)
    run = ROOT / "output" / f"gemma_dev_n300_s42_{version}"
    predictions = run / "predictions.csv"
    teacher = {}
    if predictions.exists():
        paired = read_config(run)
        if paired["cases"] != manifest or any(paired.get(k) != v for k, v in prompt.items()):
            raise ValueError("Teacher run has different inputs or prompt messages.")
        with predictions.open(newline="", encoding="utf-8") as f:
            labels = list(csv.DictReader(f))
        if [r["complaint_id"] for r in labels] != ids:
            raise ValueError("Teacher predictions do not match the frozen sample.")
        for row, entry in zip(labels, manifest, strict=True):
            if row["reference_label"] != entry["reference_label"]:
                raise ValueError(f"Teacher export reference changed for {row['complaint_id']}")
            teacher[row["complaint_id"]] = row["teacher_label"]
    cases = prepare_cases([rows[cid] for cid in ids])
    for case, frozen in zip(cases, manifest, strict=True):
        digest = hashlib.sha256(case["text"].encode()).hexdigest()
        if digest != frozen["input_sha256"] or case["reference_label"] != frozen["reference_label"]:
            raise ValueError(f"Input or reference changed for {case['complaint_id']}")
        case["teacher_label"] = teacher.get(case["complaint_id"])
    return {**config, **prompt, "teacher_run": run.name if teacher else None}, cases


@torch.inference_mode()
def predict(model, tokenizer, settings, label_ids, messages):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_tensors="pt", return_dict=True)
    prompt_length = inputs["input_ids"].shape[1]
    if prompt_length + settings.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prepared input exceeds the Student context window.")
    replies = []
    total_tokens = 0
    for attempt in range(2):
        output = model.generate(**inputs, generation_config=settings)
        tokens = output.sequences[0, prompt_length:].tolist()
        total_tokens += len(tokens)
        if tokens and tokens[-1] == tokenizer.eos_token_id:
            tokens.pop()
        raw = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        replies.append(raw)
        if attempt == 0:
            # Diagnostic only: compare the five single-letter scores at the first output position.
            choice = LABELS[output.scores[0][0, label_ids].argmax().item()]
        if raw.strip() in LABELS:
            break
    label = replies[-1].strip()
    return {
        "student_label": label if label in LABELS else "", "first_token_choice": choice,
        "first_reply": replies[0], "last_reply": replies[-1], "attempts": len(replies),
        "prompt_tokens_per_attempt": prompt_length, "generated_tokens_all_attempts": total_tokens,
    }


def score(rows, field):
    matrix = {ref: dict.fromkeys((*LABELS, "FAIL"), 0) for ref in LABELS}
    agree = 0
    for row in rows:
        label = row[field]
        matrix[row["reference_label"]][label or "FAIL"] += 1
        agree += bool(label and label == row["teacher_label"])
    paired = rows[0]["teacher_label"] is not None
    return {
        **classification_scores(matrix), "correct": sum(matrix[k][k] for k in LABELS),
        "teacher_matches": agree if paired else None,
        "teacher_agreement_all_cases": agree / len(rows) if paired else None,
        "predicted_counts": dict(Counter(row[field] or "FAIL" for row in rows)),
        "confusion_rows_reference_columns_student": matrix,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-version", choices=SUPPORTED_VERSIONS, default=DEFAULT_VERSION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config, cases = load_paired_cases(args.prompt_version)
    dirname = ("smollm2-135m-zero-shot" if args.prompt_version == "v1"
               else f"smollm2-135m-few-shot-{args.prompt_version}")
    output_dir = args.output or RESULTS / dirname
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(TOKENIZER, revision=REVISION, dtype=torch.float32).eval()
    encoded = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(tokens) != 1 for tokens in encoded):
        raise ValueError("The first-token diagnostic requires a single token for each bare letter.")
    label_ids = [tokens[0] for tokens in encoded]
    settings = GenerationConfig(do_sample=False, max_new_tokens=config["decoding"]["max_tokens"],
                                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                return_dict_in_generate=True, output_scores=True)
    rows = []
    start = time.monotonic()
    for i, case in enumerate(cases, 1):
        messages = build_messages(case["text"], config)
        result = predict(model, tokenizer, settings, label_ids, messages)
        rows.append({k: v for k, v in case.items() if k != "text"} | result)
        if i % 25 == 0:
            print(f"{i}/{len(cases)}  valid={sum(bool(r['student_label']) for r in rows)}  "
                  f"elapsed={time.monotonic()-start:.0f}s", flush=True)

    metrics = {
        "model": TOKENIZER, "revision": REVISION, "device": str(model.device), "dtype": str(model.dtype),
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "teacher_run": config["teacher_run"], "prompt": config["system_prompt"],
        "prompt_file": config["prompt_file"], "examples": config.get("examples", []),
        "final_instruction": config.get("final_instruction", ""),
        "user_template": config["user_template"], "max_new_tokens": settings.max_new_tokens,
        "do_sample": False, "batch_size": 1, "max_attempts": 2, "split": config["split"],
        "seed_sampling": config["seed"], "clipped": sum(c["clipped"] for c in cases),
        "teacher_valid": sum(bool(row["teacher_label"]) for row in rows) if config["teacher_run"] else None,
        "attempts": sum(row["attempts"] for row in rows), "inference_seconds": time.monotonic()-start,
        "strict_generation": score(rows, "student_label"),
        "first_token_choice": score(rows, "first_token_choice"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temp:
        stage = Path(temp)
        with (stage / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (stage / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        for filename in ("predictions.csv", "metrics.json"):
            (stage / filename).replace(output_dir / filename)
    for mode in ("strict_generation", "first_token_choice"):
        summary = metrics[mode]
        agreement = summary["teacher_agreement_all_cases"]
        display = f"{agreement:.3f}" if agreement is not None else "n/a (no paired Teacher run)"
        print(f"{mode}: accuracy={summary['accuracy']:.3f}, macro-F1={summary['macro_f1']:.3f}, "
              f"Teacher agreement={display}")


if __name__ == "__main__":
    main()
