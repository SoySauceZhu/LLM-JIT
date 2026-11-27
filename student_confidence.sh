#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

for size in 0100 0200 0500 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
    gemma="gemma3-270m-it_v3_n${size}"
    adapter="output/lora/$gemma"
    evaluation="${gemma}_validation"
    output="output/confidence/gemma3-270m-it-v3-lora-curve/n${size}"

    if [ -s "$output/summary.json" ] && [ -s "$output/${evaluation}.csv" ]; then
        printf 'Reusing confidence results for %s\n' "$gemma"
        continue
    fi
    if [ ! -s "$adapter/adapter_model.safetensors" ]; then
        printf 'Original Gemma adapter weights are missing at %s/adapter_model.safetensors\n' "$adapter" >&2
        exit 1
    fi

    printf 'Scoring Gemma 3 N%s\n' "$size"
    uv run --locked --extra lora python code/student_confidence.py \
        --checkpoint "output/lora_eval/$evaluation" "$adapter" \
        --device cuda --output "$output"
done
