"""Observations of Student inference; never change the tokens being generated."""

import hashlib
import json
import math

from mortgage_prompt import LABELS

DIAGNOSTICS_VERSION = 1
TOP_K = 10


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def inference_runtime(model, tokenizer, settings):
    """Record the loaded implementation and numerical controls, without secrets."""
    import torch

    parameter = next(model.parameters())
    text_config = getattr(model.config, "text_config", model.config)
    runtime = {
        "torch": torch.__version__, "torch_git_version": torch.version.git_version,
        "cuda_runtime": torch.version.cuda, "model_class": type(model).__name__,
        "device": str(parameter.device), "dtype": str(parameter.dtype),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "context_config": {name: getattr(text_config, name, None) for name in
                           ("max_position_embeddings", "sliding_window", "layer_types")},
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "threads": torch.get_num_threads(), "model_training": model.training,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(
            tokenizer.get_chat_template().encode()).hexdigest(),
        "special_token_ids": {name: getattr(tokenizer, name, None)
                              for name in ("bos_token_id", "eos_token_id", "pad_token_id")},
        "generation_config": settings.to_dict(),
    }
    if parameter.device.type == "cuda":
        properties = torch.cuda.get_device_properties(parameter.device)
        runtime["cuda"] = {
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "cudnn_version": torch.backends.cudnn.version(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            # Enabled SDPA kernels are possibilities, not a trace of the kernel used.
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        }
    return runtime


def generation_diagnostics(tokenizer, inputs, output, label_ids, stop_ids,
                           max_new_tokens, runtime):
    import torch

    prompt_ids = inputs["input_ids"][0].tolist()
    generated = output.sequences[0, len(prompt_ids):].tolist()
    removed_stop = generated[-1] if generated and generated[-1] in stop_ids else None
    first_step = None
    if output.scores:
        # Copy once; do not retain the vocabulary-sized tensors in the ledger.
        scores = output.scores[0][0].detach().float().cpu()
        nan = int(torch.isnan(scores).sum())
        positive_inf = int(torch.isposinf(scores).sum())
        negative_inf = int(torch.isneginf(scores).sum())
        normalizable = not nan and not positive_inf and bool(scores.isfinite().any())
        probabilities = scores.softmax(dim=-1) if normalizable else None

        def entry(token_id):
            value = float(scores[token_id])
            finite = bool(torch.isfinite(scores[token_id]))
            return {
                "token_id": token_id, "token": tokenizer.convert_ids_to_tokens(token_id),
                "score": value if finite else None,
                "nonfinite_score": None if finite else str(value),
                "probability": float(probabilities[token_id]) if probabilities is not None else None,
                "rank": int((scores > scores[token_id]).sum()) + 1 if not nan else None,
            }

        best_label_index = int(scores[label_ids].argmax())
        best_label_id = label_ids[best_label_index]
        best_stop_id = max(stop_ids, key=lambda i: float(scores[i])) if stop_ids and not nan else None
        margin = float(scores[best_label_id] - scores[best_stop_id]) if best_stop_id is not None else None
        first_step = {
            "score_kind": "generation_scores_after_logits_processors",
            "vocabulary_size": len(scores),
            "nonfinite_counts": {"nan": nan, "positive_inf": positive_inf, "negative_inf": negative_inf},
            "selected": entry(generated[0]) if generated else None,
            "top_tokens": [entry(i) for i in scores.topk(min(TOP_K, len(scores))).indices.tolist()],
            "labels": {label: entry(i) for label, i in zip(LABELS, label_ids)},
            "stop_tokens": [entry(i) for i in stop_ids],
            "best_label": LABELS[best_label_index],
            "best_label_minus_best_stop_score": margin if margin is not None and math.isfinite(margin) else None,
        }
    mask = inputs.get("attention_mask")
    return {
        "version": DIAGNOSTICS_VERSION, "runtime": runtime,
        "prompt": {"token_count": len(prompt_ids), "token_ids_sha256": token_hash(prompt_ids),
                   "head_token_ids": prompt_ids[:16], "tail_token_ids": prompt_ids[-32:],
                   "bos_token_count": prompt_ids.count(tokenizer.bos_token_id),
                   "attention_mask_sha256": token_hash(mask[0].tolist()) if mask is not None else None},
        "generated_token_ids": generated,
        "output_cache_class": type(output.past_key_values).__name__
            if getattr(output, "past_key_values", None) is not None else None,
        "generated_tokens": tokenizer.convert_ids_to_tokens(generated),
        "text_with_special_tokens": tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        "removed_stop_token_id": removed_stop,
        "stop_reason": "eos_token" if removed_stop is not None else
            "max_new_tokens" if len(generated) >= max_new_tokens else "other",
        "first_step": first_step,
    }
