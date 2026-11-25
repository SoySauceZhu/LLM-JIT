#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for n in 0200 1000 2000 3000 4000 5000 6000 7000 8000; do
    train_csv="data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n${n}.csv"
    output="output/lora/gemma3-270m-it_v3_n${n}"

    if [[ -e "$output" || -L "$output" ]]; then
        if [[ -s "$output/adapter_model.safetensors" && -s "$output/adapter_config.json" \
            && -s "$output/training.json" && -s "$output/loss_tokens.jsonl" ]]; then
            printf 'Already trained, skipping %s\n' "$output"
            continue
        fi
        printf 'Existing output is missing adapter files or training records: %s\n' "$output" >&2
        exit 1
    fi
    if [[ ! -s "$train_csv" ]]; then
        printf 'Training CSV is missing or empty: %s\n' "$train_csv" >&2
        exit 1
    fi

    printf 'Training %s\n' "$train_csv"
    uv run --locked --extra lora python code/student_lora.py \
        --train-csv "$train_csv" --device cuda --dtype float32 \
        --rank 8 --alpha 16 --dropout 0.05 --learning-rate 2e-4 \
        --epochs 3 --batch-size 8 --micro-batch-size 1 --seed 42 \
        --output "$output" --run
done
