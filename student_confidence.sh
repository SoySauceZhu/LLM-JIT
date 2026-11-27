#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

qwen=qwen3-0.6b_v3_n8000_fp16_mixed_mb2
gemma=gemma3-270m-it_v3_n9000_bf16_mixed_be1.5_torch28_mb4

uv run --locked --extra lora python code/student_confidence.py \
    --checkpoint "output/lora_eval/${qwen}_validation" "output/lora/$qwen" \
    --checkpoint "output/lora_eval/${gemma}_validation" "output/lora/$gemma" \
    --device cuda --output output/confidence/qwen-n8000_gemma-weighted-n9000
