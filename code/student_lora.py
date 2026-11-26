"""Train a completion-only Student LoRA from a Teacher-labeled history CSV."""

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from mortgage_data import HEAD, REVISION, TAIL, TOKENIZER, load_cohort, prepare_cases, sha256
from mortgage_prompt import LABELS, build_messages, load_prompt, parse_label
from run_store import file_sha256
from student_baseline import (
    DEFAULT_STUDENT_MODEL, LORA_STUDENT_MODELS, STUDENT_MODELS,
    adapter_identity, load_student, student_pretrained_kwargs,
)

IGNORE = -100
ANSWER_TOKENS = 2  # One category letter and the end-of-turn token.
LOSS_WINDOW = ANSWER_TOKENS + 1  # Include the position that predicts the letter.
MIXED_PRECISION_DTYPES = {"float16-mixed": "float16", "bfloat16-mixed": "bfloat16"}


@dataclass
class Example:
    complaint_id: str
    teacher_label: str
    input_ids: list[int]

    def audit(self):
        return {"complaint_id": self.complaint_id, "teacher_label": self.teacher_label,
                "input_tokens": len(self.input_ids),
                "first_supervised_index": len(self.input_ids) - ANSWER_TOKENS,
                "loss_token_ids": self.input_ids[-ANSWER_TOKENS:]}


def read_training_rows(path):
    """Allowlist the original history inputs and read only Teacher labels as targets."""
    history = {row["complaint_id"]: row for row in load_cohort() if row["split"] == "history"}
    rows = []
    skipped = 0
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"complaint_id", "split", "text", "teacher_label"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Training CSV requires columns {', '.join(sorted(required))}")
        for row in reader:
            label = parse_label(row["teacher_label"])
            if label is None:
                skipped += 1
                continue
            cid = row["complaint_id"]
            source = history.get(cid)
            if source is None or row["split"] != "history" or row["text"] != source["text"]:
                raise ValueError(f"Complaint {cid!r} does not match the frozen history input")
            if cid in seen:
                raise ValueError(f"Duplicate training complaint: {cid}")
            seen.add(cid)
            rows.append((source, label))
    if not rows:
        raise ValueError("Training CSV contains no valid history Teacher answers")
    return rows, skipped


def encode_answer(tokenizer, prompt, case, label, stop_id, context_window, *, chat_template_kwargs=None):
    """Check the training boundary against the actual generation prefix."""
    messages = build_messages(case["text"], prompt)
    kwargs = chat_template_kwargs or {}
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)["input_ids"]
    completed = [*messages, {"role": "assistant", "content": label}]
    tokens = tokenizer.apply_chat_template(completed, add_generation_prompt=False, **kwargs)["input_ids"]
    answer = tokens[len(prefix):]
    cid = case["complaint_id"]
    if not prefix or tokens[:len(prefix)] != prefix:
        raise ValueError(f"Complaint {cid}: the completed chat changed the generation prefix")

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    if (len(answer) < ANSWER_TOKENS or decode(answer[:1]) != label or answer[1] != stop_id
        or decode(answer[ANSWER_TOKENS:]).strip()):
        raise ValueError(f"Complaint {cid}: expected {label!r} and {decode([stop_id])} as the only answer tokens")
    # The template's trailing whitespace follows the stop token and has no training target.
    ids = tokens[:len(prefix) + ANSWER_TOKENS]
    if len(ids) > context_window:
        raise ValueError(f"Complaint {cid}: prepared chat exceeds the Student context window")
    return Example(cid, label, ids)


def prepare_examples(rows, tokenizer, context_window, student_model=DEFAULT_STUDENT_MODEL):
    profile = STUDENT_MODELS[student_model]
    end_of_turn, = profile["stop_tokens"]
    if end_of_turn not in tokenizer.get_vocab() or tokenizer.pad_token_id is None:
        raise ValueError(f"{student_model} tokenizer must provide {end_of_turn} and a pad token")
    stop_id = tokenizer.convert_tokens_to_ids(end_of_turn)
    prompt = load_prompt("v3")
    cases = prepare_cases([source for source, _ in rows])
    examples = [encode_answer(tokenizer, prompt, case, label, stop_id, context_window,
                              chat_template_kwargs=profile["chat_template_kwargs"])
                for case, (_, label) in zip(cases, rows, strict=True)]
    return examples, stop_id, sum(case["clipped"] for case in cases)


def collate(examples, pad_id, device):
    """Left-pad so all answer windows align, keeping all padding and prompt labels masked."""
    import torch

    length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, example in enumerate(examples):
        ids = torch.tensor(example.input_ids, dtype=torch.long)
        input_ids[index, -len(ids):] = ids
        attention_mask[index, -len(ids):] = 1
    labels = torch.full_like(input_ids, IGNORE)
    labels[:, -ANSWER_TOKENS:] = input_ids[:, -ANSWER_TOKENS:]
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    return {name: value.to(device) for name, value in {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "position_ids": position_ids, "labels": labels,
    }.items()}


def parse_class_weight(value):
    label, _, raw_weight = value.partition("=")
    try:
        weight = float(raw_weight)
    except ValueError:
        weight = 0.0
    if label not in LABELS or not 0 < weight < math.inf:
        raise argparse.ArgumentTypeError("Expected LABEL=WEIGHT with an A–E label and a positive finite weight")
    return label, weight


def normalize_class_weights(class_weights, label_counts):
    """Set the mean letter weight across the training rows to one."""
    weights = {label: class_weights.get(label, 1.0) for label in label_counts}
    scale = max(weights.values())
    scaled = {label: weight / scale for label, weight in weights.items()}
    mean = sum(label_counts[label] * weight for label, weight in scaled.items()) / sum(label_counts.values())
    return {label: weight / mean for label, weight in scaled.items()}


def completion_loss(model, batch, tokens_per_update, *, letter_weights=None):
    # Omit the LM head at other positions. The full prompt still enters the causal decoder.
    labels = batch["labels"][:, -LOSS_WINDOW:]
    if letter_weights is None:
        # The native causal LM shifts these labels internally.
        return model(**{**batch, "labels": labels}, use_cache=False, logits_to_keep=LOSS_WINDOW,
                     num_items_in_batch=tokens_per_update).loss

    from torch.nn.functional import cross_entropy

    inputs = {name: value for name, value in batch.items() if name != "labels"}
    logits = model(**inputs, use_cache=False, logits_to_keep=LOSS_WINDOW).logits
    # The last prompt token predicts the letter, and the letter predicts the end of turn.
    answer_logits = logits[:, :-1].float().flatten(0, 1)
    token_losses = cross_entropy(answer_logits, labels[:, 1:].flatten(), reduction="none").view(-1, ANSWER_TOKENS)
    return (token_losses[:, 0] * letter_weights + token_losses[:, 1]).sum() / tokens_per_update


def add_lora(base, rank, alpha, dropout, *, student_model=DEFAULT_STUDENT_MODEL, autocast_adapter_dtype=True):
    from peft import LoraConfig, TaskType, get_peft_model

    base.requires_grad_(False)
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
                        lora_dropout=dropout, target_modules=["q_proj", "v_proj"], bias="none")
    model = get_peft_model(base, config, revision=STUDENT_MODELS[student_model]["revision"],
                           autocast_adapter_dtype=autocast_adapter_dtype)
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any(not {"lora_A", "lora_B"}.intersection(name.split("."))
                            for name, _ in trainable):
        raise RuntimeError("Only LoRA A/B factors should be trainable")
    return model, [param for _, param in trainable]


def train(model, parameters, examples, pad_id, args, *, class_weights=None, sort_micro_batches_by_length=False):
    import torch

    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    amp_dtype = getattr(torch, MIXED_PRECISION_DTYPES[args.dtype]) if args.dtype in MIXED_PRECISION_DTYPES else None
    # AdamW's default epsilon rounds to zero when its moments are stored in FP16.
    optimizer_eps = 1e-4 if args.dtype == "float16" else 1e-8
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0, eps=optimizer_eps)
    scaler = torch.amp.GradScaler(args.device, enabled=amp_dtype == torch.float16)
    scale = scaler.get_scale()
    rng = random.Random(args.seed)
    order = list(range(len(examples)))
    epochs = []
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    step = 0
    skipped_updates = 0
    for epoch in range(1, args.epochs + 1):
        print(f"epoch={epoch}/{args.epochs} started", flush=True)
        rng.shuffle(order)
        loss_sum = 0.0
        for start_index in range(0, len(order), args.batch_size):
            indices = order[start_index:start_index + args.batch_size]
            if sort_micro_batches_by_length and 1 < args.micro_batch_size < len(indices):
                indices.sort(key=lambda i: len(examples[i].input_ids))
            target_count = ANSWER_TOKENS * len(indices)
            weights = None
            if class_weights is not None:
                weights = torch.tensor([class_weights[examples[i].teacher_label] for i in indices],
                                       dtype=torch.float32, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=args.device)
            for offset in range(0, len(indices), args.micro_batch_size):
                end = offset + args.micro_batch_size
                sample = [examples[i] for i in indices[offset:end]]
                batch = collate(sample, pad_id, args.device)
                letter_weights = None if weights is None else weights[offset:end]
                with torch.autocast(args.device, dtype=amp_dtype, enabled=amp_dtype is not None):
                    loss = completion_loss(model, batch, target_count, letter_weights=letter_weights)
                scaler.scale(loss).backward()
                step_loss += loss.detach()
            mean_loss = step_loss.item()
            if not math.isfinite(mean_loss):
                raise RuntimeError(f"Non-finite completion loss at optimizer update {step + 1}")
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            if new_scale < scale:
                skipped_updates += 1
            else:
                step += 1
            if new_scale == 0:
                raise RuntimeError("FP16 gradient scale reached zero")
            scale = new_scale
            loss_sum += mean_loss * target_count
        mean = loss_sum / (ANSWER_TOKENS * len(examples))
        epochs.append({"epoch": epoch, "completion_loss": mean, "optimizer_updates": step,
                       "skipped_nonfinite_updates": skipped_updates})
        print(f"epoch={epoch}/{args.epochs} optimizer_updates={step} completion_loss={mean:.6f} "
              f"skipped_nonfinite_updates={skipped_updates}", flush=True)
    if step == 0:
        raise RuntimeError("Training finished without an optimizer update")
    if not torch.stack([param.detach().isfinite().all() for param in parameters]).all().item():
        raise RuntimeError("Training produced non-finite LoRA parameters")
    if args.device == "cuda":
        torch.cuda.synchronize()
    return {"epoch_metrics": epochs, "optimizer_updates": step, "skipped_nonfinite_updates": skipped_updates,
            "optimizer_eps": optimizer_eps, "training_seconds": time.perf_counter() - start}


def save_run(output, model, examples, summary, *, student_model=DEFAULT_STUDENT_MODEL):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "adapter"
        model.save_pretrained(stage, safe_serialization=True, save_embedding_layers=False)
        summary["adapter_files_sha256"] = adapter_identity(stage, student_model)
        (stage / "training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with (stage / "loss_tokens.jsonl").open("w", encoding="utf-8") as file:
            for example in examples:
                file.write(json.dumps(example.audit()) + "\n")
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", choices=LORA_STUDENT_MODELS, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New directory for the adapter and training records")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16", *MIXED_PRECISION_DTYPES),
                        default="float32", help="Mixed modes use AMP with FP32 LoRA weights and AdamW moments")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Recompute decoder activations during backward to reduce memory use")
    parser.add_argument("--class-weight", action="append", type=parse_class_weight, default=[],
                        metavar="LABEL=WEIGHT", help="Relative Teacher class weight for letter loss, repeatable")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8, help="Effective batch size per optimizer update")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--sort-micro-batches-by-length", action="store_true",
                        help="Group similar lengths within each effective batch to reduce padding")
    parser.add_argument("--seed", type=int, default=42)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--audit-only", action="store_true", help="Check every target using the tokenizer only")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run and (args.output is None or args.output.exists()):
        parser.error("--run requires a new --output directory")
    if args.audit_only and args.output is not None:
        parser.error("--output is only used with --run")
    if min(args.rank, args.alpha, args.epochs) < 1 or not 1 <= args.micro_batch_size <= args.batch_size:
        parser.error("Rank, alpha, epochs and batch sizes must be positive, with micro batch <= effective batch")
    if not 0 <= args.dropout < 1 or not 0 < args.learning_rate < math.inf:
        parser.error("Dropout must be in [0, 1) and learning rate must be positive and finite")
    class_weights = dict(args.class_weight)
    if len(class_weights) != len(args.class_weight):
        parser.error("--class-weight may be set once per label")

    from transformers import AutoConfig, AutoTokenizer

    profile = STUDENT_MODELS[args.student_model]
    kwargs = student_pretrained_kwargs(args.student_model)
    rows, skipped = read_training_rows(args.train_csv)
    tokenizer = AutoTokenizer.from_pretrained(profile["model"], **kwargs)
    context_window = AutoConfig.from_pretrained(profile["model"], **kwargs).max_position_embeddings
    examples, stop_id, clipped = prepare_examples(rows, tokenizer, context_window, args.student_model)
    label_counts = dict(sorted(Counter(example.teacher_label for example in examples).items()))
    targets = {example.teacher_label: example.input_ids[-ANSWER_TOKENS:] for example in examples}
    print(f"examples={len(examples)} skipped_invalid={skipped} clipped={clipped} "
          f"loss_tokens={len(examples) * ANSWER_TOKENS} targets={dict(sorted(targets.items()))}", flush=True)
    loss_config = {"type": "completion_only_causal_lm_cross_entropy", "ignore_index": IGNORE,
                   "end_of_turn_id": stop_id, "supervised_tokens_per_example": ANSWER_TOKENS}
    letter_weights = None
    if class_weights:
        letter_weights = normalize_class_weights(class_weights, label_counts)
        loss_config.update(class_weights=class_weights, normalized_letter_weights=letter_weights,
                           end_of_turn_weight=1.0)
        print(f"class_weights={class_weights} normalized_letter_weights={letter_weights}", flush=True)
    if args.audit_only:
        return

    import torch

    torch.manual_seed(args.seed)
    base_dtype = MIXED_PRECISION_DTYPES.get(args.dtype, args.dtype)
    base, _, settings, _ = load_student(args.student_model, args.device, dtype=base_dtype)
    if stop_id not in settings.eos_token_id:
        raise ValueError("The supervised end-of-turn token is not a stop token for generation")
    model, parameters = add_lora(base, args.rank, args.alpha, args.dropout, student_model=args.student_model,
                                 autocast_adapter_dtype=args.dtype != "float16")
    trainable_count = sum(p.numel() for p in parameters)
    adapter_dtype = str(parameters[0].dtype)
    print(f"device={args.device} precision={args.dtype} base_dtype={base_dtype} adapter_dtype={adapter_dtype} "
          f"gradient_checkpointing={args.gradient_checkpointing} "
          f"trainable_lora_parameters={trainable_count:,}", flush=True)
    summary = {
        "student_model": profile["model"], "revision": profile["revision"], "prompt": "v3",
        "chat_template_kwargs": profile["chat_template_kwargs"],
        "chat_template_sha256": sha256(tokenizer.chat_template),
        "n_training_examples": len(examples), "n_invalid_answers_skipped": skipped, "n_clipped": clipped,
        "train_csv": str(args.train_csv), "train_csv_sha256": file_sha256(args.train_csv),
        "ordered_complaint_ids_sha256": sha256("".join(e.complaint_id + "\n" for e in examples)),
        "teacher_label_counts": label_counts,
        "clipping": {"tokenizer": TOKENIZER, "revision": REVISION, "head": HEAD, "tail": TAIL},
        "loss": loss_config,
        "device": args.device, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "precision": args.dtype, "base_dtype": base_dtype, "adapter_dtype": adapter_dtype, "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "epochs": args.epochs, "batch_size": args.batch_size, "micro_batch_size": args.micro_batch_size,
        "sort_micro_batches_by_length": args.sort_micro_batches_by_length,
        "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": 0.0,
        "trainable_parameters": trainable_count,
        "packages": {package: version(package) for package in ("peft", "torch", "transformers", "tokenizers")},
    }
    summary.update(train(model, parameters, examples, tokenizer.pad_token_id, args, class_weights=letter_weights,
                         sort_micro_batches_by_length=args.sort_micro_batches_by_length))
    save_run(args.output, model, examples, summary, student_model=args.student_model)
    print(f"Saved adapter to {args.output}", flush=True)


if __name__ == "__main__":
    main()
