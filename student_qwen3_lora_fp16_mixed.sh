#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

teacher_run=output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all
train_csv=data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n9000.csv
micro_batch_size=${QWEN_MICRO_BATCH_SIZE:-1}
case "$micro_batch_size" in
    [1-8]) ;;
    *) printf 'QWEN_MICRO_BATCH_SIZE must be an integer from 1 to 8.\n' >&2; exit 1 ;;
esac
name=qwen3-0.6b_v3_n9000_fp16_mixed
if [ "$micro_batch_size" -ne 1 ]; then
    name="${name}_mb${micro_batch_size}"
fi
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
    printf '[1/2] Training Qwen3-0.6B LoRA on 9000 samples with FP16 AMP and micro-batch size %s\n' "$micro_batch_size"
    uv run --locked --extra lora python code/student_lora.py \
        --student-model qwen3-0.6b --device cuda --dtype float16-mixed \
        --train-csv "$train_csv" --rank 8 --alpha 16 --dropout 0.05 --learning-rate 2e-4 \
        --epochs 3 --batch-size 8 --micro-batch-size "$micro_batch_size" \
        --sort-micro-batches-by-length --seed 42 --output "$adapter" --run
fi

printf '[2/2] Validating Qwen3-0.6B with the N9000 LoRA adapter\n'
uv run --locked --extra lora python code/student_baseline.py \
    --student-model qwen3-0.6b --device cuda --split validation --prompt-version v3 \
    --teacher-run "$teacher_run" --adapter "$adapter" --output "$evaluation" --run

printf 'Adapter metrics: %s/metrics.json\n' "$evaluation"
