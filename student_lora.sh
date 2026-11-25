#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

train_csv=data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n9000.csv
teacher_run=output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all
epochs=3

if [ ! -s "$train_csv" ]; then
    printf 'Training CSV is missing or empty: %s\n' "$train_csv" >&2
    exit 1
fi
if [ ! -s "$teacher_run/run.sqlite" ]; then
    printf 'Teacher ledger is missing or empty: %s/run.sqlite\n' "$teacher_run" >&2
    exit 1
fi

run_variant() {
    rank=$1
    alpha=$2
    learning_rate=$3
    name="gemma3-270m-it_v3_n9000_r${rank}_a${alpha}_lr${learning_rate}_e${epochs}_drop0.05"
    adapter="output/lora/$name"
    evaluation="output/lora_eval/${name}_validation"

    if [ -e "$adapter" ] || [ -L "$adapter" ]; then
        if [ ! -s "$adapter/adapter_model.safetensors" ] || [ ! -s "$adapter/adapter_config.json" ] \
            || [ ! -s "$adapter/training.json" ] || [ ! -s "$adapter/loss_tokens.jsonl" ]; then
            printf 'Existing output is missing adapter files or training records: %s\n' "$adapter" >&2
            exit 1
        fi
        printf 'Reusing trained adapter %s\n' "$adapter"
    else
        printf 'Training %s\n' "$name"
        uv run --locked --extra lora python code/student_lora.py \
            --train-csv "$train_csv" --device cuda --dtype float32 \
            --rank "$rank" --alpha "$alpha" --dropout 0.05 --learning-rate "$learning_rate" \
            --epochs "$epochs" --batch-size 8 --micro-batch-size 1 --seed 42 \
            --output "$adapter" --run
    fi

    printf 'Validating %s\n' "$name"
    uv run --locked --extra lora python code/student_baseline.py \
        --student-model gemma3-270m-it --device cuda --split validation --prompt-version v3 \
        --adapter "$adapter" --teacher-run "$teacher_run" --output "$evaluation" --run
}

# Compare ranks 8, 16 and 32 at alpha/rank = 2.
run_variant 8 16 5e-5
run_variant 16 32 5e-5
run_variant 32 64 5e-5

# Compare alpha/rank = 1 and 4 at each of the first two ranks.
run_variant 8 8 5e-5
run_variant 8 32 5e-5
run_variant 16 16 5e-5
run_variant 16 64 5e-5

# Check the learning rate at rank 8 and alpha 16.
run_variant 8 16 2e-5
run_variant 8 16 1e-4

# Compare alpha/rank = 1 and 2 at rank 32 as well.
run_variant 32 32 5e-5
