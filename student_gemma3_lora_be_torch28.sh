#!/bin/sh
# Temporary launcher for a Linux x86_64 CUDA image with a PyTorch 2.8 limit.
set -eu

CDPATH= cd "$(dirname "$0")"

teacher_run=output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all
train_csv=data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n9000.csv
environment=.cache/venvs/torch28-cu126
python="$environment/bin/python"
micro_batch_size=${GEMMA_MICRO_BATCH_SIZE:-4}
case "$micro_batch_size" in
    [1-8]) ;;
    *) printf 'GEMMA_MICRO_BATCH_SIZE must be an integer from 1 to 8.\n' >&2; exit 1 ;;
esac
name=gemma3-270m-it_v3_n9000_bf16_mixed_be1.5_torch28
if [ "$micro_batch_size" -ne 1 ]; then
    name="${name}_mb${micro_batch_size}"
fi
adapter="output/lora/$name"
evaluation="output/lora_eval/${name}_validation"

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    printf 'This launcher requires Linux x86_64 with an NVIDIA CUDA GPU.\n' >&2
    exit 1
fi
if [ ! -s "$train_csv" ] || [ ! -s "$teacher_run/run.sqlite" ]; then
    printf 'Expected training data at %s and Teacher ledger at %s/run.sqlite.\n' "$train_csv" "$teacher_run" >&2
    exit 1
fi

if [ ! -x "$python" ]; then
    uv venv --python 3.12 "$environment"
fi
overrides="$environment/overrides.txt"
printf '%s\n' 'torch==2.8.0+cu126' > "$overrides"
uv pip install --python "$python" --no-sources --torch-backend cu126 \
    --requirements pyproject.toml --extra lora --overrides "$overrides"

"$python" - <<'PY'
import torch

if torch.__version__ != "2.8.0+cu126":
    raise SystemExit(f"Expected PyTorch 2.8.0+cu126, found {torch.__version__}")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
    raise SystemExit("CUDA and native BF16 must be available to the current process")
x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
if (x @ x)[0, 0].item() != 16:
    raise SystemExit("BF16 CUDA smoke test failed")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name()}")
PY

if [ -e "$adapter" ] || [ -L "$adapter" ]; then
    if [ ! -s "$adapter/adapter_model.safetensors" ] || [ ! -s "$adapter/adapter_config.json" ] \
        || [ ! -s "$adapter/training.json" ] || [ ! -s "$adapter/loss_tokens.jsonl" ]; then
        printf 'Existing output is missing adapter files or training records: %s\n' "$adapter" >&2
        exit 1
    fi
    printf '[1/2] Reusing trained adapter %s\n' "$adapter"
else
    printf '[1/2] Training Gemma 3 270M IT with BF16 AMP and micro-batch size %s\n' "$micro_batch_size"
    "$python" code/student_lora.py \
        --student-model gemma3-270m-it --device cuda --dtype bfloat16-mixed \
        --train-csv "$train_csv" --class-weight B=1.5 --class-weight E=1.5 \
        --rank 8 --alpha 16 --dropout 0.05 --learning-rate 2e-4 \
        --epochs 3 --batch-size 8 --micro-batch-size "$micro_batch_size" \
        --sort-micro-batches-by-length --seed 42 --output "$adapter" --run
fi

printf '[2/2] Running October validation\n'
"$python" code/student_baseline.py \
    --student-model gemma3-270m-it --device cuda --split validation --prompt-version v3 \
    --teacher-run "$teacher_run" --adapter "$adapter" --output "$evaluation" --run

printf 'Adapter metrics: %s/metrics.json\n' "$evaluation"
