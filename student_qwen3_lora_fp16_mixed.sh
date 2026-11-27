#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

teacher_run=output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all
train_csv=data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n9000.csv
adapter=output/lora/qwen3-0.6b_v3_n9000_fp16_mixed
evaluation=output/lora_eval/qwen3-0.6b_v3_n9000_fp16_mixed_validation

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
    printf '[1/2] Training Qwen3-0.6B LoRA on 9000 history samples with mixed precision\n'
    uv run --locked --extra lora python code/student_lora.py \
        --student-model qwen3-0.6b --device cuda --dtype float16-mixed --gradient-checkpointing \
        --train-csv "$train_csv" --rank 8 --alpha 16 --dropout 0.05 --learning-rate 2e-4 \
        --epochs 3 --batch-size 8 --micro-batch-size 1 --seed 42 --output "$adapter" --run
fi

printf '[2/2] Validating Qwen3-0.6B with the N9000 LoRA adapter\n'
uv run --locked --extra lora python code/student_baseline.py \
    --student-model qwen3-0.6b --device cuda --split validation --prompt-version v3 \
    --teacher-run "$teacher_run" --adapter "$adapter" --output "$evaluation" --run

printf 'Adapter metrics: %s/metrics.json\n' "$evaluation"
