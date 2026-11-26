#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

teacher_run=output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all
train_csv=data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n9000.csv
base_eval=output/baseline_v3/student_qwen3-0.6b_validation_cuda
adapter=output/lora/qwen3-0.6b_v3_n9000
adapter_eval=output/lora_eval/qwen3-0.6b_v3_n9000_validation

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
fi

validate() {
    eval_output=$1
    shift
    uv run --locked --extra lora python code/student_baseline.py \
        --student-model qwen3-0.6b --device cuda --split validation --prompt-version v3 \
        --teacher-run "$teacher_run" --output "$eval_output" "$@" --run
}

printf '[1/3] Validating the original Qwen3-0.6B\n'
validate "$base_eval"

if [ -d "$adapter" ]; then
    printf '[2/3] Reusing trained adapter %s\n' "$adapter"
else
    printf '[2/3] Training Qwen3-0.6B LoRA on 9000 history samples\n'
    uv run --locked --extra lora python code/student_lora.py \
        --student-model qwen3-0.6b --device cuda --dtype float32 --gradient-checkpointing --train-csv "$train_csv" \
        --rank 8 --alpha 16 --dropout 0.05 --learning-rate 2e-4 \
        --epochs 3 --batch-size 8 --micro-batch-size 1 --seed 42 --output "$adapter" --run
fi

printf '[3/3] Validating Qwen3-0.6B with the N9000 LoRA adapter\n'
validate "$adapter_eval" --adapter "$adapter"

printf 'Baseline metrics: %s/metrics.json\nAdapter metrics: %s/metrics.json\n' "$base_eval" "$adapter_eval"
