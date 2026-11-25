"""Preview, run or export mortgage Student baselines on a chronological partition."""

import argparse
import json
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

from evaluation import score
from mortgage_data import COHORT_SPLIT_COUNTS, REVISION, TOKENIZER, default_output, load_experiment, prepare_cases, select_rows
from mortgage_prompt import DEFAULT_VERSION, LABELS, ROOT, SUPPORTED_VERSIONS, build_messages, load_prompt, parse_label
from run_store import (
    attempts_by_case, file_sha256, final_attempt, initialize, make_config, progress_metrics, read_config,
    single_runner, usage_metrics, write_exports,
)
from teacher_client import DEFAULT_TEACHER_MODEL, TEACHER_MODELS, teacher_output
from student_diagnostics import DIAGNOSTICS_VERSION, TOP_K, generation_diagnostics, inference_runtime

DEFAULT_STUDENT_MODEL = "gemma3-270m-it"
STUDENT_MODELS = {
    "smollm2-135m": {"model": TOKENIZER, "revision": REVISION, "chat_template_kwargs": {}},
    "qwen3.5-0.8b": {"model": "Qwen/Qwen3.5-0.8B",
                     "revision": "2fc06364715b967f1860aea9cf38778875588b17",
                     "chat_template_kwargs": {"enable_thinking": False}},
    "gemma3-270m-it": {"model": "google/gemma-3-270m-it",
                       "revision": "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
                       "chat_template_kwargs": {}, "stop_tokens": ["<end_of_turn>"]},
}


def student_config(cases, split, prompt, student_model=DEFAULT_STUDENT_MODEL, device="cpu", *,
                   diagnostics=False, attn_implementation="auto", adapter=None):
    profile = STUDENT_MODELS[student_model]
    extra = {"generation_stop_tokens": profile["stop_tokens"]} if "stop_tokens" in profile else {}
    if diagnostics:
        extra["diagnostics"] = {"version": DIAGNOSTICS_VERSION, "first_step_top_k": TOP_K}
    if attn_implementation != "auto":
        extra["attn_implementation"] = attn_implementation
    if adapter is not None:
        extra["lora_adapter"] = adapter
    return make_config(cases, split, prompt, backend="student", model=profile["model"],
                       decoding={"do_sample": False, "max_tokens": 16,
                                 "eos_policy": "model_and_tokenizer"}, device=device,
                       dtype="torch.float32", threads=4, batch_size=1, max_attempts=2,
                       student_model=student_model, model_revision=profile["revision"],
                       inference_tokenizer=profile["model"], inference_tokenizer_revision=profile["revision"],
                       chat_template_kwargs=profile["chat_template_kwargs"], **extra)


def student_output(student_model, split, version):
    # Preserve the existing SmolLM2 path while isolating additional models.
    backend = "student" if student_model == "smollm2-135m" else f"student_{student_model}"
    return default_output(backend, split, version)


def generation_stop_ids(model_eos, tokenizer_eos, extra_stop_ids=()):
    """Stop on both end-of-text and the native chat end-of-turn markers."""
    model_ids = model_eos if isinstance(model_eos, (list, tuple)) else [model_eos]
    return list(dict.fromkeys(n for n in [tokenizer_eos, *model_ids, *extra_stop_ids]
                              if n is not None)) or None


def student_pretrained_kwargs(student_model):
    kwargs = {"revision": STUDENT_MODELS[student_model]["revision"]}
    if student_model != "smollm2-135m":
        kwargs["cache_dir"] = ROOT / ".cache/huggingface"
    return kwargs


def adapter_identity(directory, student_model):
    profile = STUDENT_MODELS[student_model]
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text())
    if (config.get("peft_type"), config.get("task_type"), config.get("base_model_name_or_path"),
        config.get("revision")) != ("LORA", "CAUSAL_LM", profile["model"], profile["revision"]):
        raise ValueError("Adapter must be a causal LM LoRA for the pinned Student model and revision")
    return {name: file_sha256(directory / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")}


def load_student(student_model, device="cpu", *, attn_implementation="auto", dtype="float32", adapter=None):
    """Load the native text model/tokenizer from an immutable revision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    profile = STUDENT_MODELS[student_model]
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Metal GPU is unavailable in this process; use --device cpu or a GPU-enabled execution environment")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA GPU is unavailable in this process; check nvidia-smi, install CUDA-enabled "
            "PyTorch with uv sync --locked on Linux, or use --device cpu"
        )
    if dtype == "bfloat16" and (device != "cuda" or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError("Native bfloat16 training requires a supported CUDA GPU; use --dtype float32")
    kwargs = student_pretrained_kwargs(student_model)
    torch.set_num_threads(4)
    try:
        tokenizer = AutoTokenizer.from_pretrained(profile["model"], **kwargs)
        model_kwargs = {"attn_implementation": attn_implementation} if attn_implementation != "auto" else {}
        model = AutoModelForCausalLM.from_pretrained(
            profile["model"], dtype=getattr(torch, dtype), **kwargs, **model_kwargs).to(device).eval()
    except OSError as exc:
        if student_model == "gemma3-270m-it" and "gated" in str(exc).lower():
            raise RuntimeError(
                "Gemma model access is required. Review the conditions at "
                "https://huggingface.co/google/gemma-3-270m-it and configure an authorized "
                "HF_TOKEN or run `hf auth login` locally, then resume the same command."
            ) from exc
        raise
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    encoded = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(tokens) != 1 for tokens in encoded):
        raise ValueError("First-token diagnostic requires single-token letters")
    extra_stop_ids = []
    for token in profile.get("stop_tokens", ()):
        if token not in tokenizer.get_vocab():
            raise ValueError(f"Native chat stop token is missing from the tokenizer: {token}")
        extra_stop_ids.append(tokenizer.convert_tokens_to_ids(token))
    eos = generation_stop_ids(model.generation_config.eos_token_id, tokenizer.eos_token_id,
                              extra_stop_ids)
    settings = GenerationConfig(do_sample=False, max_new_tokens=16, eos_token_id=eos,
                                pad_token_id=tokenizer.pad_token_id,
                                return_dict_in_generate=True, output_scores=True)
    return model, tokenizer, settings, [tokens[0] for tokens in encoded]


def load_teacher_labels(run, config, *, required=False, teacher_model=None):
    """Read the Teacher ledger, including legacy pilots, without modifying it."""
    run = Path(run)
    if not (run / "run.sqlite").is_file():
        if required:
            raise ValueError(f"Teacher ledger not found: {run}")
        return None, "missing"
    teacher = read_config(run)
    if teacher.get("backend", "teacher") != "teacher":
        raise ValueError("--teacher-run must point to a Teacher ledger")
    if teacher_model is not None and teacher["model"] != TEACHER_MODELS[teacher_model]["model"]:
        raise ValueError("Teacher model differs from --teacher-model")
    for key in ("tokenizer", "revision", "head", "tail", "system_prompt", "user_template", "prompt_file"):
        if teacher.get(key) != config.get(key):
            raise ValueError(f"Teacher configuration mismatch: {key}")
    for key, empty in (("examples", []), ("final_instruction", "")):
        if teacher.get(key, empty) != config.get(key, empty):
            raise ValueError(f"Teacher prompt mismatch: {key}")
    if teacher["decoding"]["max_tokens"] != config["decoding"]["max_tokens"]:
        raise ValueError("Teacher output budget differs")
    expected, actual = config["cases"], teacher["cases"]
    if teacher.get("schema_version") == 3 and teacher["split"] is None:
        # An all-cohort Teacher ledger supplies only the selected Student partition.
        actual = [case for case in actual if case["split"] == config["split"]]
    elif teacher.get("split") != config.get("split"):
        raise ValueError("Teacher configuration mismatch: split")
    identity = ("complaint_id", "reference_label", "input_sha256", "clipped", "student_tokens_before_clip")
    if teacher.get("schema_version") == 3 and config.get("schema_version") == 3:
        identity += ("split",)
    if len(expected) != len(actual) or any(
            any(a.get(key) != b.get(key) for key in identity) for a, b in zip(expected, actual)):
        raise ValueError("Teacher inputs, order or references differ from the Student manifest")
    uri = (run.resolve() / "run.sqlite").as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        grouped = attempts_by_case(db)
    labels = {}
    for case in actual:
        final = final_attempt(grouped.get(case["complaint_id"], []))
        if final is None:
            return None, "incomplete"
        labels[case["complaint_id"]] = final["label"] or ""
    return labels, "complete"


def predict_once(model, tokenizer, settings, label_ids, messages, chat_template_kwargs=None, *,
                 diagnostics=False, runtime=None):
    import torch

    with torch.inference_mode():
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_tensors="pt", return_dict=True,
                                               **(chat_template_kwargs or {}))
        device = getattr(model, "device", None)
        if isinstance(device, (str, torch.device)):
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        max_positions = getattr(model.config, "max_position_embeddings", None)
        if max_positions is None:
            max_positions = model.config.text_config.max_position_embeddings
        if prompt_length + settings.max_new_tokens > max_positions:
            raise ValueError("Prepared input exceeds the Student context window")
        output = model.generate(**inputs, generation_config=settings)
        tokens = output.sequences[0, prompt_length:].tolist()
        generated_tokens = len(tokens)
        eos = getattr(settings, "eos_token_id", None)
        eos = tokenizer.eos_token_id if eos is None else eos
        stop_ids = eos if isinstance(eos, (list, tuple)) else [eos]
        if tokens and tokens[-1] in stop_ids:
            tokens.pop()
        raw = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        choice = LABELS[output.scores[0][0, label_ids].argmax().item()] if output.scores else None
    result = {"raw_reply": raw, "first_token_choice": choice,
              "prompt_tokens": prompt_length, "completion_tokens": generated_tokens}
    if diagnostics:
        result["diagnostics"] = generation_diagnostics(
            tokenizer, inputs, output, label_ids, stop_ids, settings.max_new_tokens, runtime)
    return result


def collect_student(db, cases, predict):
    """Commit each attempt; a crash between retries resumes with the remaining budget."""
    grouped = attempts_by_case(db)
    for index, case in enumerate(cases, 1):
        attempts = grouped.get(case["complaint_id"], [])
        if final_attempt(attempts):
            continue
        used = sum(a["status"] != "paused" for a in attempts)
        for _ in range(used, 2):
            start = time.monotonic()
            try:
                result = predict(case)
            except Exception as exc:
                with db:
                    db.execute("""INSERT INTO attempts (complaint_id, status, detail, seconds)
                        VALUES (?, 'paused', ?, ?)""",
                               (case["complaint_id"], str(exc), time.monotonic() - start))
                raise
            raw = result["raw_reply"]
            label = parse_label(raw)
            with db:
                fields = ["complaint_id", "status", "raw_reply", "label", "prompt_tokens",
                          "completion_tokens", "seconds", "first_token_choice"]
                values = [case["complaint_id"], "ok" if label else "invalid", raw, label,
                          result["prompt_tokens"], result["completion_tokens"],
                          time.monotonic() - start, result["first_token_choice"]]
                if "diagnostics" in result:
                    fields.append("diagnostics")
                    values.append(json.dumps(result["diagnostics"], allow_nan=False))
                db.execute(f"INSERT INTO attempts ({', '.join(fields)}) "
                           f"VALUES ({', '.join('?' for _ in values)})", values)
            if label:
                break
        if index % 25 == 0 or index == len(cases):
            print(f"{index}/{len(cases)}", flush=True)


def export_results(db, output, teacher_run, *, require_teacher=False, teacher_model=None):
    config = json.loads(db.execute("SELECT config FROM run WHERE id=1").fetchone()[0])
    labels, pairing_status = load_teacher_labels(teacher_run, config, required=require_teacher,
                                                teacher_model=teacher_model)
    teacher = read_config(teacher_run) if (Path(teacher_run) / "run.sqlite").is_file() else None
    grouped = attempts_by_case(db)
    fields = ["complaint_id", "split", "reference_label", "sequence_no", "received_at_utc", "arrival_index",
              "input_sha256", "student_tokens_before_clip", "clipped", "teacher_label",
              "student_label", "status", "first_token_choice", "first_reply", "last_reply",
              "attempts", "prompt_tokens_all_attempts", "generated_tokens_all_attempts", "seconds"]
    rows, terminal = [], []
    for case in config["cases"]:
        attempts = grouped.get(case["complaint_id"], [])
        generated = [a for a in attempts if a["status"] != "paused"]
        final = final_attempt(attempts)
        row = {field: case.get(field) for field in fields}
        row.update(teacher_label=labels[case["complaint_id"]] if labels is not None else None,
                   student_label=final["label"] if final else None,
                   status=final["status"] if final else "pending",
                   first_token_choice=generated[0]["first_token_choice"] if generated else None,
                   first_reply=generated[0]["raw_reply"] if generated else None,
                   last_reply=generated[-1]["raw_reply"] if generated else None,
                   attempts=len(attempts),
                   prompt_tokens_all_attempts=sum(a["prompt_tokens"] or 0 for a in attempts),
                   generated_tokens_all_attempts=sum(a["completion_tokens"] or 0 for a in attempts),
                   seconds=sum(a["seconds"] for a in attempts))
        rows.append(row)
        if final:
            terminal.append(row)
    progress = progress_metrics(len(rows), len(terminal))
    # Never publish an all-cases agreement from a partially completed run.
    paired = labels is not None and progress["complete"]
    metrics = {**progress, "teacher_run": str(teacher_run.resolve()) if labels is not None else None,
               "teacher_model": teacher["model"] if teacher else None,
               "model": config["model"], "model_revision": config.get("model_revision", config["revision"]),
               "device": config["device"], "dtype": config["dtype"],
               "pairing_status": pairing_status, "clipped": sum(c["clipped"] for c in config["cases"]),
               "usage": usage_metrics(db),
               "strict_generation": score(terminal, "student_label", paired=paired),
               "first_token_choice": score(terminal, "first_token_choice", paired=paired)}
    json_files = {"config.json": config, "metrics.json": metrics}
    if config.get("diagnostics"):
        diagnostic_attempts = [
            {"attempt_id": a["id"], "complaint_id": cid, "attempt_number": index,
             "status": a["status"], "raw_reply": a["raw_reply"],
             "recorded_at": a["recorded_at"],
             "diagnostics": json.loads(a["diagnostics"]) if a.get("diagnostics") else None}
            for cid, attempts in grouped.items() for index, a in enumerate(attempts, 1)
        ]
        diagnostic_attempts.sort(key=lambda a: a["attempt_id"])
        metrics["diagnostics"] = {
            "version": DIAGNOSTICS_VERSION, "file": "diagnostics.json",
            "attempts_recorded": sum(a["diagnostics"] is not None for a in diagnostic_attempts),
            "generation_attempts": sum(a["status"] != "paused" for a in diagnostic_attempts),
        }
        json_files["diagnostics.json"] = {"version": DIAGNOSTICS_VERSION,
                                          "attempts": diagnostic_attempts}
    write_exports(output, json_files=json_files,
                  csv_files={"predictions.csv": (fields, rows)})
    print(f"terminal={len(terminal)}/{len(rows)} complete={progress['complete']} "
          f"accuracy={metrics['strict_generation']['accuracy']} pairing={pairing_status} "
          f"output={output}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", choices=STUDENT_MODELS, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu",
                        help="Inference device: CPU, Apple Metal, or NVIDIA CUDA (all FP32)")
    parser.add_argument("--split", choices=COHORT_SPLIT_COUNTS, default="validation")
    parser.add_argument("--prompt-version", choices=SUPPORTED_VERSIONS, default=DEFAULT_VERSION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adapter", type=Path, help="Load a saved Gemma 3 LoRA for generation")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Record generated token IDs, first-step scores and inference runtime per attempt")
    parser.add_argument("--attn-implementation", choices=("auto", "eager", "sdpa"), default="auto",
                        help="Attention implementation; auto preserves the model/library default")
    parser.add_argument("--teacher-run", type=Path)
    parser.add_argument("--teacher-model", choices=TEACHER_MODELS,
                        help="Teacher to pair against (default: DeepSeek; an explicit --teacher-run selects its stored model)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.adapter is not None and (args.student_model != "gemma3-270m-it" or args.output is None):
        parser.error("--adapter requires --student-model gemma3-270m-it and a separate --output")
    if args.export_only and args.adapter is not None:
        parser.error("--export-only uses the adapter identity already stored in the ledger")
    selected_teacher = args.teacher_model or (None if args.teacher_run else DEFAULT_TEACHER_MODEL)
    output = args.output or student_output(args.student_model, args.split, args.prompt_version)
    if args.export_only:
        config = read_config(output)
        if config.get("schema_version") not in (2, 3) or config.get("backend") != "student":
            parser.error("Export requires a schema 2 or 3 Student baseline ledger")
        version = Path(config["prompt_file"]).stem.rsplit("_", 1)[-1]
        teacher_run = args.teacher_run or teacher_output(selected_teacher, None, version)
        with single_runner(output), closing(sqlite3.connect(output / "run.sqlite")) as db:
            export_results(db, output, teacher_run, require_teacher=args.teacher_run is not None,
                           teacher_model=selected_teacher)
        return
    rows = select_rows(load_experiment(), args.split)
    print(f"{args.split} n={len(rows)} counts={dict(Counter(r['reference_label'] for r in rows))} "
          f"model={args.student_model} prompt={args.prompt_version} output={output}")
    if not args.run:
        print("Preview only. Add --run to start.")
        return
    prompt = load_prompt(args.prompt_version)
    cases = prepare_cases(rows)
    config = student_config(cases, args.split, prompt, args.student_model, args.device,
                            diagnostics=args.diagnostics, attn_implementation=args.attn_implementation,
                            adapter=adapter_identity(args.adapter, args.student_model) if args.adapter else None)
    teacher_run = args.teacher_run or teacher_output(selected_teacher, None, args.prompt_version)
    # Detect a wrong Teacher configuration before spending local inference time.
    load_teacher_labels(teacher_run, config, required=args.teacher_run is not None,
                        teacher_model=selected_teacher)
    output.mkdir(parents=True, exist_ok=True)
    with single_runner(output), closing(sqlite3.connect(output / "run.sqlite")) as db:
        initialize(db, config)
        try:
            grouped = attempts_by_case(db)
            if any(not final_attempt(grouped.get(c["complaint_id"], [])) for c in cases):
                model, tokenizer, settings, label_ids = load_student(
                    args.student_model, args.device, attn_implementation=args.attn_implementation,
                    adapter=args.adapter)
                runtime = inference_runtime(model, tokenizer, settings) if args.diagnostics else None
                collect_student(db, cases, lambda case: predict_once(
                    model, tokenizer, settings, label_ids, build_messages(case["text"], prompt),
                    config["chat_template_kwargs"], diagnostics=args.diagnostics, runtime=runtime))
        finally:
            export_results(db, output, teacher_run, require_teacher=args.teacher_run is not None,
                           teacher_model=selected_teacher)


if __name__ == "__main__":
    main()
