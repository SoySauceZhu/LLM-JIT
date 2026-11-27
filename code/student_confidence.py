"""Score the first answer token for archived Student validation checkpoints."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from mortgage_data import load_experiment, prepare_cases, select_rows
from mortgage_prompt import LABELS, build_messages, load_prompt
from run_store import file_sha256, read_config, write_exports
from student_baseline import LORA_STUDENT_MODELS, adapter_identity, load_student, student_config


def load_validation(run, adapter, cases, prompt):
    config = read_config(run)
    student = config["student_model"]
    if student not in LORA_STUDENT_MODELS or config["split"] != "validation":
        raise ValueError(f"Expected an October Qwen3 or Gemma 3 validation ledger: {run}")
    expected = student_config(cases, "validation", prompt, student)
    for key in (*prompt, "cases", "data_sha256", "backend", "model", "model_revision",
                "inference_tokenizer_revision", "chat_template_kwargs", "dtype"):
        if config.get(key) != expected[key]:
            raise ValueError(f"Frozen validation configuration differs for {key}: {run}")
    identity = adapter_identity(adapter, student) if adapter is not None else None
    if identity != config.get("lora_adapter"):
        raise ValueError(f"Adapter does not match the checkpoint used for validation: {run}")
    predictions = run / "predictions.csv"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if not metrics["complete"] or metrics["pairing_status"] != "complete":
        raise ValueError(f"Expected completed validation paired with the Teacher: {run}")
    with predictions.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for case, row in zip(cases, rows, strict=True):
        if any(str(case[key]) != row[key] for key in ("complaint_id", "input_sha256", "reference_label", "split")):
            raise ValueError(f"Validation predictions differ from the input manifest: {run}")
        if row["status"] not in ("ok", "invalid") or any(
                row[key] and row[key] not in LABELS for key in ("student_label", "teacher_label", "first_token_choice")):
            raise ValueError(f"Invalid archived prediction record for {row['complaint_id']}: {run}")
    return config, rows


def label_confidence(logits, label_ids):
    """Measure the five-way distribution and the vocabulary mass of those labels."""
    import torch

    logits = logits.float()
    scores = logits[label_ids]
    probabilities = scores.softmax(dim=-1)
    mass = logits.softmax(dim=-1)[label_ids].sum()
    values = torch.cat((scores, probabilities, mass.reshape(1))).tolist()
    if not all(map(math.isfinite, values)):
        raise ValueError("Non-finite first-token logits or probabilities")
    scores = values[:len(LABELS)]
    probabilities = values[len(LABELS):-1]
    best = max(range(len(LABELS)), key=scores.__getitem__)
    return {
        "five_way_label": LABELS[best],
        "confidence": probabilities[best],
        "label_mass": values[-1],
        **dict(zip((f"logit_{label}" for label in LABELS), scores, strict=True)),
        **dict(zip((f"p_{label}" for label in LABELS), probabilities, strict=True)),
    }


def score_checkpoint(config, adapter, cases, archived, prompt, device):
    import torch

    model, tokenizer, _, label_ids = load_student(
        config["student_model"], device, adapter=adapter,
        attn_implementation=config.get("attn_implementation", "auto"))
    results = []
    with torch.inference_mode():
        for index, (case, row) in enumerate(zip(cases, archived, strict=True), 1):
            inputs = tokenizer.apply_chat_template(
                build_messages(case["text"], prompt), add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True, **config["chat_template_kwargs"])
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
            result = label_confidence(logits, label_ids)
            results.append({
                "complaint_id": row["complaint_id"],
                "reference_label": row["reference_label"],
                "teacher_label": row["teacher_label"],
                "strict_label": row["student_label"],
                "strict_label_probability": result.get(f"p_{row['student_label']}"),
                "archived_first_token_choice": row["first_token_choice"],
                **result,
            })
            if index % 50 == 0 or index == len(cases):
                print(f"{config['student_model']} {index}/{len(cases)}", flush=True)
    return results


def summarize(rows):
    correct = [row["confidence"] for row in rows if row["five_way_label"] == row["reference_label"]]
    wrong = [row["confidence"] for row in rows if row["five_way_label"] != row["reference_label"]]
    bins = []
    for lower, upper in zip((0.0, 0.4, 0.6, 0.8, 0.9, 0.95), (0.4, 0.6, 0.8, 0.9, 0.95, 1.0), strict=True):
        group = [row for row in rows if lower <= row["confidence"] and (row["confidence"] < upper or upper == 1.0)]
        count = sum(row["five_way_label"] == row["reference_label"] for row in group)
        bins.append({"lower": lower, "upper": upper, "n": len(group), "cfpb_correct": count,
                     "cfpb_accuracy": count / len(group) if group else None})
    return {
        "n": len(rows),
        "five_way_cfpb_accuracy": len(correct) / len(rows),
        "archived_strict_cfpb_accuracy": sum(r["strict_label"] == r["reference_label"] for r in rows) / len(rows),
        "five_way_mean_confidence_correct": mean(correct) if correct else None,
        "five_way_mean_confidence_wrong": mean(wrong) if wrong else None,
        "archived_strict_invalid": sum(not r["strict_label"] for r in rows),
        "archived_first_token_choices_matched": sum(r["five_way_label"] == r["archived_first_token_choice"] for r in rows),
        "confidence_bins": bins,
    }


REVIEW_FIELDS = ("complaint_id", "reference_label", "teacher_label", "model_1_confidence", "model_2_confidence",
                 "min_confidence", "model_1_label_mass", "model_2_label_mass")


def reference_review(first, second):
    """Rank shared strict disagreements with CFPB that also agree with the Teacher."""
    review = []
    for a, b in zip(first, second, strict=True):
        for key in ("complaint_id", "reference_label", "teacher_label"):
            if a[key] != b[key]:
                raise ValueError(f"Checkpoints differ on {key} for {a['complaint_id']}")
        teacher = a["teacher_label"]
        if teacher and teacher != a["reference_label"] and (
                a["strict_label"] == b["strict_label"] == a["five_way_label"] == b["five_way_label"] == teacher):
            strengths = a["confidence"], b["confidence"]
            review.append(dict(zip(REVIEW_FIELDS,
                                   (a["complaint_id"], a["reference_label"], teacher, *strengths, min(strengths),
                                    a["label_mass"], b["label_mass"]), strict=True)))
    return sorted(review, key=lambda row: row["min_confidence"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("VALIDATION_RUN", "ADAPTER"),
                        required=True, help="One or two archived runs. Use '-' for an unadapted model.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    args = parser.parse_args()
    checkpoints = [(Path(run), None if adapter == "-" else Path(adapter)) for run, adapter in args.checkpoint]
    if len(checkpoints) > 2 or len({run.name for run, _ in checkpoints}) != len(checkpoints):
        parser.error("Provide one or two validation runs with distinct directory names")
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output requires a new directory")

    prompt = load_prompt("v3")
    cases = prepare_cases(select_rows(load_experiment(), "validation"))
    sources = [load_validation(run, adapter, cases, prompt) for run, adapter in checkpoints]
    if len(sources) == 2 and any(a["teacher_label"] != b["teacher_label"]
                                 for a, b in zip(sources[0][1], sources[1][1], strict=True)):
        parser.error("The checkpoints use different Teacher labels")
    summary = {"split": "validation", "device": args.device, "dtype": "float32", "prompt": "v3",
               "definition": "Maximum softmax over the raw first-token logits for A through E",
               "checkpoints": []}
    exports = {}
    results = []
    for (run, adapter), (config, archived) in zip(checkpoints, sources, strict=True):
        rows = score_checkpoint(config, adapter, cases, archived, prompt, args.device)
        name = f"{run.name}.csv"
        exports[name] = (tuple(rows[0]), rows)
        summary["checkpoints"].append({
            "validation_run": str(run), "predictions_sha256": file_sha256(run/"predictions.csv"),
            "model_revision": config["model_revision"], "adapter": config.get("lora_adapter"),
            "confidence_file": name, **summarize(rows),
        })
        results.append(rows)
    if len(results) == 2:
        review = reference_review(*results)
        summary["reference_review_count"] = len(review)
        exports["reference_review.csv"] = (REVIEW_FIELDS, review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp) / "confidence"
        stage.mkdir()
        write_exports(stage, json_files={"summary.json": summary}, csv_files=exports)
        stage.rename(args.output)
    print(f"Confidence results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
