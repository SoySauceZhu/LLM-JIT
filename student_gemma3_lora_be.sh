#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

teacher_run=output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all
train_csv=data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n9000.csv
name=gemma3-270m-it_v3_n9000_bf16_mixed_be1.5
adapter="output/lora/$name"
evaluation="output/lora_eval/${name}_validation"

if [ ! -s "$train_csv" ]; then
    printf 'Training CSV is missing or empty: %s\n' "$train_csv" >&2
    exit 1
fi
if [ ! -s "$teacher_run/run.sqlite" ]; then
    printf 'Teacher ledger is missing or empty: %s/run.sqlite\n' "$teacher_run" >&2
    exit 1
fi

if [ -e "$adapter" ] || [ -L "$adapter" ]; then
    if [ ! -s "$adapter/adapter_model.safetensors" ] || [ ! -s "$adapter/adapter_config.json" ] \
        || [ ! -s "$adapter/training.json" ] || [ ! -s "$adapter/loss_tokens.jsonl" ]; then
        printf 'Existing output is missing adapter files or training records: %s\n' "$adapter" >&2
        exit 1
    fi
    printf '[1/2] Reusing trained adapter %s\n' "$adapter"
else
    printf '[1/2] Training Gemma 3 270M IT LoRA with BF16 AMP and B/E class weights\n'
    uv run --locked --extra lora python code/student_lora.py \
        --student-model gemma3-270m-it --device cuda --dtype bfloat16-mixed \
        --train-csv "$train_csv" --class-weight B=1.5 --class-weight E=1.5 \
        --rank 8 --alpha 16 --dropout 0.05 --learning-rate 2e-4 \
        --epochs 3 --batch-size 8 --micro-batch-size 1 --seed 42 --output "$adapter" --run
fi

printf '[2/2] Validating Gemma 3 270M IT with the B/E-weighted adapter\n'
uv run --locked --extra lora python code/student_baseline.py \
    --student-model gemma3-270m-it --device cuda --split validation --prompt-version v3 \
    --teacher-run "$teacher_run" --adapter "$adapter" --output "$evaluation" --run

printf 'Adapter metrics: %s/metrics.json\n' "$evaluation"
