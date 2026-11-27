"""Train and evaluate a frozen English BGE + multinomial LR Student."""

import argparse
import json
import re
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import mortgage_embedding as encoder
from confidence_gated_fallback import read_csv, require, write_csv
from embedding_lr_results import save_json, write_results
from evaluation import score, teacher_agreement_by_class
from mortgage_data import DATA, load_cohort, select_rows, sha256
from mortgage_prompt import LABELS, ROOT, load_prompt
from run_store import file_sha256, single_runner


TRAIN_DIR = ROOT / "data/lora/teacher_deepseek-v4.1-flash-1017_v3"
OUTPUT = ROOT / "result/validation/bge-small-en-v1.5-v3-embedding-lr"
CACHE = ROOT / ".cache/embedding_lr"
SIZES = (0, 100, 200, 500, *range(1000, 9001, 1000))
LR_OPTIONS = {"solver": "lbfgs", "C": 10.0, "fit_intercept": True,
              "class_weight": None, "max_iter": 2000, "tol": 1e-8}
PROTOTYPE_SCALE = 10.0


def load_inputs():
    manifest_path = TRAIN_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = manifest["source"]
    teacher_path = ROOT / source["teacher_predictions_csv"]
    prompt_path = ROOT / source["prompt"]
    hashes = {}
    for path, expected in ((DATA, source["cohort_csv_sha256"]),
                           (teacher_path, source["teacher_predictions_csv_sha256"]),
                           (prompt_path, source["prompt_sha256"]),
                           *((TRAIN_DIR / p["file"], p["csv_sha256"]) for p in manifest["prefixes"])):
        digest = file_sha256(path)
        require(digest == expected, f"Frozen artifact changed: {path}")
        hashes[str(path.relative_to(ROOT))] = digest
    hashes[str(manifest_path.relative_to(ROOT))] = file_sha256(manifest_path)
    cohort = load_cohort()
    history = select_rows(cohort, "history")
    validation = select_rows(cohort, "validation")
    teacher = {}
    for row in read_csv(teacher_path):
        if row["split"] in {"history", "validation"}:
            require(row["complaint_id"] not in teacher, "Duplicate Teacher ID")
            teacher[row["complaint_id"]] = row
    for row in (*history, *validation):
        archived = teacher.get(row["complaint_id"])
        metadata = ("sequence_no", "split", "received_at_utc", "reference_label")
        require(archived is not None and all(row[key] == archived[key] for key in metadata),
                f"Teacher metadata mismatch: {row['complaint_id']}")
        require(archived["teacher_label"] in (*LABELS, "") and
                (archived["status"] == "ok") == bool(archived["teacher_label"]), "Invalid Teacher status/label")
    history = [(r, teacher[r["complaint_id"]]["teacher_label"]) for r in history
               if teacher[r["complaint_id"]]["teacher_label"]][:SIZES[-1]]
    for n in SIZES[1:]:
        frozen = read_csv(TRAIN_DIR / f"history_n{n:04d}.csv")
        require(len(frozen) == n, f"Wrong n{n} prefix size")
        for (case, label), row in zip(history[:n], frozen, strict=True):
            require((case["complaint_id"], case["text"], label) ==
                    (row["complaint_id"], row["text"], row["teacher_label"]), f"n{n} prefix mismatch")
    validation = [{**r, "teacher_label": teacher[r["complaint_id"]]["teacher_label"],
                   "teacher_status": teacher[r["complaint_id"]]["status"],
                   "teacher_attempts": int(teacher[r["complaint_id"]]["attempts"]),
                   "teacher_input_sha256": teacher[r["complaint_id"]]["input_sha256"]} for r in validation]
    descriptions = dict(re.findall(r"^([A-E]) = (.+?)(?=\n\n)", load_prompt("v3")["system_prompt"], re.M | re.S))
    require(tuple(descriptions) == LABELS, "Expected the five v3 category descriptions")
    descriptions = {label: " ".join(description.split()) for label, description in descriptions.items()}
    return history, validation, descriptions, source["teacher_model"], hashes


def get_embeddings(texts, settings, device, cache):
    cache.mkdir(parents=True, exist_ok=True)
    identity = {**settings, "input_sha256": sha256("".join(sha256(text) for text in texts))}
    digest = sha256(json.dumps(identity, sort_keys=True))
    path = cache / f"{digest}.npz"
    reused = path.is_file()
    if not reused:
        started = time.perf_counter()
        vectors, token_counts, chunks, chunk_size = encoder.encode_texts(
            texts, device=device, batch_size=settings["batch_size"])
        with TemporaryDirectory(dir=cache) as temporary:
            temporary_path = Path(temporary) / path.name
            np.savez(temporary_path, vectors=vectors, token_counts=token_counts, chunks=chunks,
                     chunk_size=chunk_size, seconds=time.perf_counter() - started)
            temporary_path.replace(path)
    with np.load(path, allow_pickle=False) as data:
        vectors, token_counts = data["vectors"], data["token_counts"]
        stats = {key: data[key].item() for key in ("chunks", "chunk_size", "seconds")}
    require(vectors.shape == (len(texts), encoder.DIMENSION) and token_counts.shape == (len(texts),)
            and (token_counts > 0).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6),
            f"Invalid embedding cache {path}")
    return vectors, token_counts, {**identity, **stats, "cache_key": digest,
                                   "cache_reused": reused, "cache_sha256": file_sha256(path)}


def fit_head(n, features, targets, prototypes):
    if n == 0:
        return {"kind": "zero_shot_category_prototypes", "scale": PROTOTYPE_SCALE,
                "coef": (PROTOTYPE_SCALE * prototypes.astype(np.float64)).tolist(),
                "intercept": [0.0] * len(LABELS), "fit_iterations": 0}
    model = LogisticRegression(**LR_OPTIONS)
    model.fit(features[:n], targets[:n])
    require(tuple(model.classes_) == LABELS and model.n_iter_[0] < LR_OPTIONS["max_iter"],
            f"n{n} missing a class or failed to converge")
    return {"kind": "multinomial_logistic_regression", "options": LR_OPTIONS,
            "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
            "fit_iterations": int(model.n_iter_[0])}


def predict_head(features, head):
    logits = features.astype(np.float64) @ np.asarray(head["coef"]).T + np.asarray(head["intercept"])
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def infer(n, validation, probabilities, token_counts):
    rows = []
    for case, p, tokens in zip(validation, probabilities, token_counts, strict=True):
        label = LABELS[int(p.argmax())]
        rows.append({"n_train": n, **{k: case[k] for k in ("complaint_id", "split", "sequence_no",
                     "received_at_utc", "reference_label", "teacher_label", "teacher_status", "teacher_attempts",
                     "text_sha256", "teacher_input_sha256")}, "embedding_tokens": int(tokens),
                     "five_way_label": label, "confidence": float(p.max()),
                     **{f"p_{c}": float(value) for c, value in zip(LABELS, p, strict=True)}})
    return rows


def run(args, device, stage):
    history, validation, descriptions, teacher_name, hashes = load_inputs()
    packages = {name: version(name) for name in ("torch", "transformers", "tokenizers", "huggingface-hub",
                                                "numpy", "scipy", "scikit-learn", "matplotlib", "threadpoolctl")}
    gpu = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    settings = {"model": encoder.MODEL, "revision": encoder.REVISION, "dimension": encoder.DIMENSION,
                "max_length": encoder.MAX_LENGTH, "pooling": "normalized CLS, content-token-weighted sum, L2 norm",
                "device": str(device), "gpu": gpu.name if gpu else None,
                "gpu_memory_bytes": gpu.total_memory if gpu else None,
                "cuda_runtime": torch.version.cuda if gpu else None,
                "dtype": "float16" if gpu else "float32", "batch_size": args.batch_size,
                "encoder_sha256": file_sha256(Path(encoder.__file__)),
                "embedding_versions": {name: packages[name] for name in ("torch", "transformers", "tokenizers")}}
    train_texts = [case["text"] for case, _ in history]
    texts = [*train_texts, *(case["text"] for case in validation), *descriptions.values()]
    vectors, counts, embedding = get_embeddings(texts, settings, device, args.cache)
    print(f"Embeddings ready ({embedding['seconds']:.2f}s, cache reused={embedding['cache_reused']})", flush=True)
    n_history = len(history)
    end_validation = n_history + len(validation)
    config = {"split": "validation", "sizes": SIZES, "teacher": teacher_name, "labels": LABELS,
              "packages": packages, "source_sha256": hashes, "embedding": embedding,
              "code_sha256": {name: file_sha256(ROOT / "code" / name) for name in (
                  "student_embedding_lr.py", "mortgage_embedding.py", "embedding_lr_results.py",
                  "confidence_gated_fallback.py", "evaluation.py")},
              "descriptions": descriptions, "lr_options": LR_OPTIONS, "prototype_scale": PROTOTYPE_SCALE,
              "history_multichunk_cases": int((counts[:n_history] > embedding["chunk_size"]).sum()),
              "validation_multichunk_cases": int((counts[n_history:end_validation] > embedding["chunk_size"]).sum()),
              "validation_max_content_tokens": int(counts[n_history:end_validation].max())}
    targets = [label for _, label in history]
    student_rows, checkpoint_scores = {}, {}
    with threadpool_limits(limits=4, user_api="blas"):
        for n in SIZES:
            started = time.perf_counter()
            head = fit_head(n, vectors[:n_history], targets, vectors[end_validation:])
            fit_seconds = time.perf_counter() - started if n else 0.0
            label_counts = Counter(targets[:n])
            head.update(n_train=n, classes=LABELS, embedding_model=encoder.MODEL,
                        embedding_revision=encoder.REVISION, feature_dimension=encoder.DIMENSION,
                        training_label_counts={c: label_counts[c] for c in LABELS}, fit_seconds=fit_seconds)
            save_json(stage / "models" / f"n{n:04d}.json", head)
            p = predict_head(vectors[n_history:end_validation], head)
            rows = infer(n, validation, p, counts[n_history:end_validation])
            student_rows[n] = rows
            checkpoint_scores[n] = {**score(rows, "five_way_label", paired=True), "fit_seconds": fit_seconds,
                                    "fit_iterations": head["fit_iterations"],
                                    "teacher_agreement_by_label": teacher_agreement_by_class(rows, "five_way_label"),
                                    "training_label_counts": head["training_label_counts"]}
            print(f"Scored n{n}: accuracy={checkpoint_scores[n]['accuracy']:.4f}", flush=True)
    save_json(stage / "config.json", config)
    write_csv(stage / "predictions.csv", [r for rows in student_rows.values() for r in rows])

    write_results(stage, config, student_rows, checkpoint_scores, score(validation, "teacher_label", paired=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require(args.batch_size > 0, "Batch size must be positive")
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA was requested but is unavailable")
    require(args.device != "mps" or torch.backends.mps.is_available(), "MPS was requested but is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    with single_runner(args.output), TemporaryDirectory(dir=args.output) as temporary:
        stage = Path(temporary)
        run(args, torch.device(args.device), stage)
        # A note is the completion marker. Publish it after the data and figures.
        for path in sorted(stage.rglob("*"), key=lambda p: p.name == "note.md"):
            if path.is_file():
                target = args.output / path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)


if __name__ == "__main__":
    main()
