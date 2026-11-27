#!/bin/sh
set -eu

CDPATH= cd "$(dirname "$0")"

teacher_run=output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all
train_dir=data/lora/teacher_deepseek-v4.1-flash-1017_v3
micro_batch_size=${QWEN_MICRO_BATCH_SIZE:-1}
case "$micro_batch_size" in
    [1-8]) ;;
    *) printf 'QWEN_MICRO_BATCH_SIZE must be an integer from 1 to 8.\n' >&2; exit 1 ;;
esac
suffix=
if [ "$micro_batch_size" -ne 1 ]; then
    suffix="_mb${micro_batch_size}"
fi

if [ "$#" -eq 0 ]; then
    set -- 100 200 500 1000 2000 3000 4000 5000 6000 7000 8000 9000
fi
if [ ! -s "$teacher_run/run.sqlite" ]; then
    printf 'Teacher ledger is missing or empty: %s/run.sqlite\n' "$teacher_run" >&2
    exit 1
fi

for samples in "$@"; do
    case "$samples" in
        100|200|500|[1-9]000) ;;
        *)
            printf 'Unknown training size %s. Use 100, 200, 500, or 1000 through 9000 in steps of 1000.\n' \
                "$samples" >&2
            exit 1
            ;;
    esac
    prefix=$(printf '%04d' "$samples")
    train_csv="$train_dir/history_n${prefix}.csv"
    if [ ! -s "$train_csv" ]; then
        printf 'Training CSV is missing or empty: %s\n' "$train_csv" >&2
        exit 1
    fi
done

total=$#
index=0
for samples in "$@"; do
    index=$((index + 1))
    prefix=$(printf '%04d' "$samples")
    train_csv="$train_dir/history_n${prefix}.csv"
    name="qwen3-0.6b_v3_n${prefix}_fp16_mixed${suffix}"
    adapter="output/lora/$name"
    evaluation="output/lora_eval/${name}_validation"
    printf '[%s/%s] Qwen3-0.6B N%s, FP16 AMP, micro-batch %s\n' "$index" "$total" "$prefix" "$micro_batch_size"

    if [ -e "$adapter" ] || [ -L "$adapter" ]; then
        if [ ! -s "$adapter/adapter_model.safetensors" ] || [ ! -s "$adapter/adapter_config.json" ] \
            || [ ! -s "$adapter/training.json" ] || [ ! -s "$adapter/loss_tokens.jsonl" ]; then
            printf 'Existing output is missing adapter files or training records: %s\n' "$adapter" >&2
            exit 1
        fi
        printf 'Reusing trained adapter %s\n' "$adapter"
    else
        uv run --locked --extra lora python code/student_lora.py \
            --student-model qwen3-0.6b --device cuda --dtype float16-mixed \
            --train-csv "$train_csv" --rank 8 --alpha 16 --dropout 0.05 --learning-rate 2e-4 \
            --epochs 3 --batch-size 8 --micro-batch-size "$micro_batch_size" \
            --sort-micro-batches-by-length --seed 42 --output "$adapter" --run
    fi

    printf 'Validating N%s on October complaints\n' "$prefix"
    uv run --locked --extra lora python code/student_baseline.py \
        --student-model qwen3-0.6b --device cuda --split validation --prompt-version v3 \
        --teacher-run "$teacher_run" --adapter "$adapter" --output "$evaluation" --run
    printf 'Adapter metrics: %s/metrics.json\n' "$evaluation"
done
