#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

for n in 0100 0200 0500 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
    adapter="output/lora/gemma3-270m-it_v3_n${n}"
    output="output/lora_eval/gemma3-270m-it_v3_n${n}_validation"

    if [ ! -s "$adapter/adapter_config.json" ] || [ ! -s "$adapter/adapter_model.safetensors" ]; then
        printf 'Missing adapter config or weights in %s\n' "$adapter" >&2
        exit 1
    fi

    printf 'Validating %s\n' "$adapter"
    uv run --locked --extra lora python code/student_baseline.py \
        --student-model gemma3-270m-it --device cuda --split validation --prompt-version v3 \
        --adapter "$adapter" \
        --teacher-run output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all \
        --output "$output" --run
done
