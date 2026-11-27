"""Frozen BGE representations for complaint narratives and category descriptions."""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIMENSION = 384
MAX_LENGTH = 512


def encode_texts(texts, *, device, batch_size):
    """Cover the whole narrative with disjoint chunks, padding by batch length."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    chunk_size = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)
    tokenized = tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    chunks = []
    for owner, tokens in enumerate(tokenized):
        if not tokens:
            raise ValueError(f"Empty tokenized input at index {owner}")
        for start in range(0, len(tokens), chunk_size):
            content = tokens[start:start + chunk_size]
            chunks.append((owner, len(content), [tokenizer.cls_token_id, *content, tokenizer.sep_token_id]))
    chunks.sort(key=lambda chunk: chunk[1])
    vectors = np.zeros((len(texts), DIMENSION), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            inputs = tokenizer.pad([{"input_ids": ids} for _, _, ids in batch],
                                   pad_to_multiple_of=8, return_tensors="pt").to(device)
            cls = model(**inputs).last_hidden_state[:, 0, :].float()
            unit = torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy()
            for (owner, token_count, _), vector in zip(batch, unit, strict=True):
                vectors[owner] += token_count * vector
            if start == 0 or (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(chunks):
                print(f"Encoded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks", flush=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Non-finite or zero embedding norm")
    return vectors / norms, np.array([len(tokens) for tokens in tokenized]), len(chunks), chunk_size
