#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

gemma=gemma3-270m-it_v3_n9000_bf16_mixed_be1.5_torch28_mb4

uv run --locked --extra lora python code/student_confidence.py \
    --checkpoint "output/lora_eval/${gemma}_validation" "output/lora/$gemma" \
    --device cuda --output output/confidence/gemma-weighted-n9000
