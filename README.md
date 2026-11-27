# LLM-JIT

[System and experiment design](https://docs.google.com/document/d/1LyIxZPZP4Ax1GHQ-LLyzqF2JipfCaxy5SB8kU0B53ug/edit?usp=sharing) · [Local experiment design](Experiment%20Design.md)

The default pairing is **DeepSeek V4.1 Flash 1017 Teacher → Gemma 3 270M IT Student**, using the frozen v3 prompt. Qwen3-0.6B is also available for LoRA training. Gemma 4 31B IT remains an optional Teacher baseline, and SmolLM2-135M and Qwen3.5-0.8B remain available for Student baseline inference.

[Student accuracy 与 Teacher consistency 未达 90% 的综合分析](Student_Accuracy_Consistency_Analysis.md)汇总任务难度、标签口径、类别不平衡、训练方案、输出失败及 confidence 证据，并区分已确认结论与待验证原因。

## Environment

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) from the repository root. `.python-version` selects Python 3.12.10; `pyproject.toml` and `uv.lock` pin direct and transitive dependencies.

```bash
uv sync --locked
uv run --locked python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

uv creates an isolated `.venv`; no activation is required. The supported targets are Apple Silicon macOS (PyPI PyTorch 2.10.0 for CPU/MPS) and Linux x86_64 (official PyTorch 2.10.0+cu126). The NVIDIA driver must support CUDA 12.6; uv does not install the driver. The platform-specific source configuration follows [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

The former environment is retained in the git-ignored `.venv-pip-backup/`. Recreate environments on each machine rather than copying them across platforms. `requirements.txt` has been replaced by the uv manifest and lockfile. Use `uv add PACKAGE` to change dependencies, verify with the offline tests, and commit both manifest and lockfile. Transformers 5.5.4 uses the pinned compatible huggingface-hub 1.11.0.

Teacher calls require `SILICONFLOW_API_KEY` in the repository `.env` or environment. Gemma Student weights require an authorized Hugging Face account:

```bash
uv run --locked hf auth login
```

Installing dependencies does not download model weights. Previews do not load tokenizers or models and make no API calls. A real Teacher run downloads only the shared clipping tokenizer if needed; Teacher weights stay with the API provider. Student runs download their pinned weights/tokenizers on demand. Keep credentials out of Git.

## Data: exactly three splits

`data/mortgage_complaint.csv` contains **12,196** unique complaints. Both `load_cohort()` and `load_experiment()` return this complete, audited cohort.

| Split | Receipt window | Complaints | Later use |
|---|---|---:|---|
| `history` | January–September | 9,159 | Teacher-labeled training history |
| `validation` | October | 1,013 | Model comparison and selection |
| `test` | November–December | 2,024 | Final evaluation |

The former January partition (957 rows) is now part of `history`. Only those CSV split fields changed; IDs, dates, text, reference labels and row order are preserved. The ordered-ID SHA-256 remains `3c59aa321b65ea493b579cd503af0aeb580fa9c28ee4cabe4b15d136ab123dfa`; the full CSV byte hash changed and is recorded in each new run. Loading verifies counts, time boundaries, unique IDs/text hashes and text integrity.

There is no fourth split or separate pilot loader in the current interface. The old split name and the former `--size`/`--seed` sampling options are rejected. Archived 300-case runs keep their original configurations and remain historical evidence; they are not rewritten to pretend they used the new partitioning.

## Teacher: label the full cohort

Preview all 12,196 complaints:

```bash
uv run --locked python code/teacher_baseline.py --prompt-version v3
```

Collect all labels with the default DeepSeek Teacher:

```bash
uv run --locked python code/teacher_baseline.py \
  --prompt-version v3 \
  --concurrency 1 \
  --run
```

The default destination is `output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all/`. Omitting `--split` means the complete cohort; the run configuration stores `split: null`, and every case retains its actual `history`, `validation` or `test` value. `all` in the directory name is a collection scope, not a fourth data split.

An optional `--split history`, `--split validation` or `--split test` filters to that complete partition; destinations use `teacher_<profile>_<split>/`. Concurrency defaults to 1 and may be raised for any selection; exported rows always follow the frozen input order, even if responses finish out of order. Per-attempt completion timestamps are retained. Use 1 for a serial collection. Concurrency alone does not enforce TPM; check the selected model's actual account quota. HTTP 429 still pauses and preserves the run.

Both Teachers use the existing `SILICONFLOW_API_KEY` and `https://api.siliconflow.com/v1`, with temperature 0 and max_tokens 16:

| `--teacher-model` | API model | Request settings |
|---|---|---|
| `deepseek-v4.1-flash-1017` (default) | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `enable_thinking=false`, passed through OpenAI SDK `extra_body` |
| `gemma4-31b-it` | `google/gemma-4-31B-it` | Original Gemma request settings, without the DeepSeek thinking parameter |

The dated DeepSeek ID was verified against the international account's model list. The thinking switch follows the [SiliconFlow Chat Completions API](https://docs.siliconflow.com/en/api-reference/chat-completions/chat-completions); no DeepSeek official-site API key is required. The selected API model and DeepSeek request body are frozen in the ledger; returned model IDs and response IDs are recorded on every response. Both models receive the same v3 instructions, five demonstrations and complaint. CFPB references are not sent. A legal A–E reply is accepted even when it differs from the reference. Narrative clipping stays fixed to the SmolLM2 tokenizer: above 3,000 tokens, retain the first 2,000 and last 1,000.

Run the optional Gemma Teacher baseline in its own default directory:

```bash
uv run --locked python code/teacher_baseline.py \
  --teacher-model gemma4-31b-it --prompt-version v3 --concurrency 1 --run
```

This writes `output/baseline_v3/teacher_gemma4-31b-it_all/`. Running the default command does not reuse an older unqualified `teacher_all/` directory. To resume an existing schema-3 Gemma run there with unchanged inputs/environment, explicitly select Gemma and supply `--output output/baseline_v3/teacher_all`. Switching models in any existing run directory is rejected before new API calls; keep their labels separate.

This command only records labels and attempts. It does not create training checkpoints, prefix manifests, training jobs or adapters. The current source CSV and manifest retain CFPB references for auditing/scoring; any future trainer must use only accepted Teacher labels as supervision and allowlist history IDs.

For one real connectivity check on the first history complaint:

```bash
uv run --locked python code/teacher_smoke.py --prompt-version v3

# Optional Gemma connectivity check.
uv run --locked python code/teacher_smoke.py --teacher-model gemma4-31b-it --prompt-version v3
```

## Ledger, resume and outputs

Every new run uses schema 3 and writes:

| File | Contents |
|---|---|
| `run.sqlite` | `run`: frozen config and input manifest; `attempts`: every request/generation attempt, raw reply, parsed label, status, errors, timings and usage |
| `config.json` | Frozen settings and ordered input manifest, including each case's real split; no duplicate narrative |
| `predictions.csv` | One row per selected complaint, including pending/failed rows and split |
| `metrics.json` | Model identity, progress, validity, reference-label scores and usage; Student exports also identify their paired Teacher model; aggregate all-cohort scores are collection diagnostics, not a validation/test comparison |

New ledgers contain only the `run` and `attempts` business tables. No `history_traces`, `prefixes` or `progress` tables, `history_labels.csv` or prefix JSON files are created. A separate `runner.lock.sqlite` prevents simultaneous writers; it holds no classification or training records.

Each attempt is committed immediately. Accepted labels and exhausted failures are determined from attempts, which also provide all information needed to select training subsets later. Reports are exported when collection finishes or exits normally. A timeout or invalid Teacher reply gets one retry; other API errors, including HTTP 429, pause the run. Respect a recorded `Retry-After` value before resuming. Paused service errors do not consume the classification retry budget. A Student runtime error also pauses rather than becoming a classification failure.

Rerun the same command in the same environment to resume; completed complaints are skipped, including exhausted failures. On Ctrl+C, Teacher collection stops scheduling new requests, drains in-flight requests, saves and exports. A hard kill during an uncommitted call may cause a repeated API charge. A fully complete run needs no API credentials when rerun. Token usage is recorded, but dollar costs stay unpriced until rates are supplied.

Changing the source snapshot, device, model, prompt or frozen environment requires a new `--output` directory. Schema 2 ledgers cannot resume into schema 3 and are not migrated or modified automatically. Their existing exports remain readable; basic export-only operations support schemas 2 and 3 without rewriting the stored ledger or regenerating old training-prefix artifacts.

```bash
uv run --locked python code/teacher_baseline.py \
  --export-only --output output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all
```

## Student baselines and Teacher pairing

Student defaults to `--student-model gemma3-270m-it` and the complete `validation` partition. The other Student options are `smollm2-135m`, `qwen3-0.6b` and `qwen3.5-0.8b`. `--split history` and `--split test` select their respective complete partitions. Run final test scoring only after model selection is frozen.

```bash
# Preview; no model download or inference.
uv run --locked python code/student_baseline.py \
  --student-model gemma3-270m-it --split validation

# Remote NVIDIA validation run.
nvidia-smi
uv run --locked python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
uv run --locked python code/student_baseline.py \
  --student-model gemma3-270m-it \
  --device cuda --split validation --prompt-version v3 \
  --output output/baseline_v3/student_gemma3-270m-it_validation_cuda \
  --run
```

Use `--device mps` for Apple Metal or `--device cpu`; all devices use FP32, four CPU threads, batch size 1, greedy decoding and at most 16 generated tokens per attempt. Model and inputs move to the selected device. Unavailable accelerators cause an error before Student weight/tokenizer loading, with no silent CPU fallback. Use separate output directories when changing devices. The CUDA integration has offline fixture coverage and a completed archived 300-case remote run.

To investigate empty replies, run a fresh validation experiment with diagnostics:

```bash
uv run --locked python code/student_baseline.py \
  --student-model gemma3-270m-it \
  --device cuda --split validation --prompt-version v3 \
  --diagnostics \
  --output output/baseline_v3/student_gemma3-270m-it_validation_cuda_run3 \
  --run
```

`--diagnostics` records each attempt's complete generated token IDs, text before removing the terminal stop token, stop reason, prompt-token hash and boundaries, and the first generation step's top 10 candidates, all A–E candidates and all stop tokens. Candidates include scores, full-vocabulary probabilities and ranks; nonfinite scores are explicitly recorded. The score margin is the best letter's score minus the best stop token's score. The runtime record includes the actual attention implementation, generation settings, tokenizer template hash, PyTorch/CUDA versions, GPU identity and numerical controls. Enabled SDPA backends are recorded as possibilities, not as a trace of the CUDA kernel actually used.

Diagnostics are committed as JSON in the same `attempts` row as the reply and exported to `diagnostics.json` on finish or interruption. Both retries are retained. `--export-only` regenerates the file from the ledger without inference. Diagnostics do not change prompts, generation settings or label parsing, and require no extra model forward passes; copying and summarizing scores adds runtime overhead. Continue using the same flags to resume run3. Existing runs remain readable and do not gain retroactive diagnostics; enabling diagnostics requires a new output directory.

The default `--attn-implementation auto` preserves the model/library default. To compare against ordinary attention, add `--attn-implementation eager` and use another directory, for example `output/baseline_v3/student_gemma3-270m-it_validation_cuda_eager`. Changing the implementation is a separate experiment; run3 above only adds observation.

Student automatically reads `output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all/`. Use `--teacher-model gemma4-31b-it` to select the optional Teacher's default directory, or `--teacher-run PATH` to select a specific ledger, including the older Gemma `teacher_all/`. An explicit path selects its stored Teacher model unless `--teacher-model` is also supplied, in which case the model must match. There is no automatic fallback from a missing DeepSeek run to Gemma. `metrics.json` records the actual paired Teacher API model.

An all-cohort Teacher ledger is filtered by the Student's real split, then matched by ordered IDs, input hashes, reference labels, clipping and exact prompt settings. Only that partition must have terminal Teacher outcomes; unrelated pending partitions do not block pairing. A single-partition Teacher ledger must match the complete selected inputs exactly. Archived Teacher ledgers are read-only and remain usable with matching archived Student manifests. Pairing is an export/evaluation choice and does not alter frozen Student inference settings or force another inference run.

An absent default Teacher ledger or incomplete selected Teacher partition leaves agreement null. A missing explicit path or incompatible manifest is an error. Accepted legal labels count toward Teacher agreement; exhausted failures count as nonmatches. Only terminal Student cases enter classification scores, with pending cases excluded and `complete=false`/`provisional=true` until completion. All five classes are included in macro-F1. The first-token A–E diagnostic is reported separately and never replaces a failed generated reply.

Re-export after Teacher completion without rerunning Student inference:

```bash
uv run --locked python code/student_baseline.py \
  --export-only \
  --output output/baseline_v3/student_gemma3-270m-it_validation_cuda \
  --teacher-model deepseek-v4.1-flash-1017

# Recompute agreement against the optional Gemma Teacher using the same Student predictions.
uv run --locked python code/student_baseline.py \
  --export-only \
  --output output/baseline_v3/student_gemma3-270m-it_validation_cuda \
  --teacher-model gemma4-31b-it
```

| Student | Model | Pinned revision |
|---|---|---|
| `gemma3-270m-it` | `google/gemma-3-270m-it` | `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |

Each Student uses its native tokenizer/chat template while narrative clipping stays fixed to SmolLM2. Gemma's template folds the system instruction into the first user turn and generation also stops at native `<end_of_turn>`. Qwen uses `enable_thinking=False`, the text-only backbone, and both tokenizer/model EOS IDs. Only the terminal stop token is removed before strict parsing. Qwen/Gemma caches use the git-ignored `.cache/huggingface/`; SmolLM2 uses the default Hugging Face cache.

## Student LoRA SFT on NVIDIA CUDA

From the repo root on the NVIDIA machine, audit one history prefix with the pinned tokenizer, then train a fresh adapter in a new directory.

```bash
TRAIN_CSV=data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n0100.csv
ADAPTER_DIR=output/lora/gemma3-270m-it_v3_n0100

uv run --locked --extra lora python code/student_lora.py --train-csv "$TRAIN_CSV" --audit-only
uv run --locked --extra lora python code/student_lora.py \
  --train-csv "$TRAIN_CSV" --device cuda --output "$ADAPTER_DIR" --run
```

The trainer uses the frozen v3 chat and shared narrative clipping. It accepts only complaint IDs and text from the frozen history cohort and skips rows without a valid A–E `teacher_label`. The only unmasked targets are the final assistant letter and the native turn terminator (`<end_of_turn>` for Gemma and `<|im_end|>` for Qwen3). Qwen3 training and inference both use `enable_thinking=False`. By default, each optimizer update gives all its answer tokens equal weight, including when the final effective batch is partial.

Defaults are FP32, rank 8, alpha 16, dropout 0.05 on `q_proj` and `v_proj`, AdamW at `2e-4`, three epochs, effective batch 8, micro-batch 1 and seed 42. Use `--dtype bfloat16` on a CUDA GPU with native BF16 support. The output contains PEFT adapter weights, `training.json` with runtime and optimizer metrics, and `loss_tokens.jsonl` with each sample's supervised token IDs and zero-based target start.

- `--dtype float16-mixed` stores the frozen base in FP16 and uses FP16 autocast, dynamic loss scaling, and FP32 LoRA parameters and AdamW moments. Scaling is updated after each effective batch and skipped updates are recorded in `training.json`.
- `--dtype bfloat16-mixed` uses a BF16 base and BF16 autocast on supported CUDA GPUs. LoRA parameters and AdamW moments are FP32.
- `--dtype float16` uses FP16 base weights, LoRA parameters, gradients, and AdamW moments. AdamW uses `eps=1e-4`, which is representable in FP16. Loss cross-entropy uses FP32 in every precision mode.

Evaluate the saved adapter with the same autoregressive generation and strict label parsing as the baseline. Use a fresh output directory for each adapter.

```bash
uv run --locked --extra lora python code/student_baseline.py \
  --device cuda --split validation --prompt-version v3 --adapter "$ADAPTER_DIR" \
  --output output/lora_eval/gemma3-270m-it_v3_n0100_validation --run
```

### Gemma 3 N9000 with B/E weighting

Run `sh student_gemma3_lora_be.sh` on the RTX 4090 to train with BF16 mixed precision and gradient checkpointing disabled, then run FP32 October validation. The script uses the original N9000 learning rate of `2e-4` and writes metrics to `output/lora_eval/gemma3-270m-it_v3_n9000_bf16_mixed_be1.5_validation/metrics.json`.

The repeatable `--class-weight LABEL=WEIGHT` option applies to the letter target for the corresponding Teacher class. The launcher sets B and E to `1.5` relative to the default `1` for other classes. The trainer normalizes the mean letter weight to `1` across the accepted training rows, gives every turn terminator weight `1`, and divides the weighted loss across all answer tokens in each effective batch. The requested and normalized weights appear in `training.json`.

### Qwen3-0.6B N9000 comparison

For a smoke run, use the 100-row prefix above with `--student-model qwen3-0.6b` and a separate adapter directory.

Run Qwen3 with and without LoRA on the complete October validation partition. The training command uses the same N9000 prefix, rank 8, alpha 16, learning rate `2e-4`, three epochs and FP32 defaults as the original Gemma 3 N9000 run.

Run all three steps with `sh student_qwen3_lora.sh`, or run them individually below. Qwen3 training uses gradient checkpointing, which saves GPU memory by recomputing decoder activations during backpropagation.

```bash
TEACHER_RUN=output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all
TRAIN_CSV=data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n9000.csv
ADAPTER_DIR=output/lora/qwen3-0.6b_v3_n9000

uv run --locked --extra lora python code/student_baseline.py \
  --student-model qwen3-0.6b --device cuda --split validation --prompt-version v3 \
  --teacher-run "$TEACHER_RUN" --output output/baseline_v3/student_qwen3-0.6b_validation_cuda --run
uv run --locked --extra lora python code/student_lora.py \
  --student-model qwen3-0.6b --device cuda --gradient-checkpointing \
  --train-csv "$TRAIN_CSV" --output "$ADAPTER_DIR" --run
uv run --locked --extra lora python code/student_baseline.py \
  --student-model qwen3-0.6b --device cuda --split validation --prompt-version v3 \
  --teacher-run "$TEACHER_RUN" --adapter "$ADAPTER_DIR" \
  --output output/lora_eval/qwen3-0.6b_v3_n9000_validation --run
```

Use `strict_generation.teacher_agreement_all_cases` to compare the two Students on the full cohort. `teacher_agreement_by_class` uses each valid Teacher label as its denominator and counts Student failures as disagreements. Each run also records accuracy and macro-F1 against the CFPB reference labels.

Run the FP16 mixed-precision Qwen3 learning curve with `sh student_qwen3_lora_fp16_mixed.sh`. It trains the 100, 200, and 500 sample prefixes and each 1,000-sample increment through 9,000, with gradient checkpointing disabled and an FP32 October validation after each adapter. Pass sizes as arguments to select specific prefixes, for example `sh student_qwen3_lora_fp16_mixed.sh 100 200 500`. Set `QWEN_MICRO_BATCH_SIZE=2` to use micro-batch 2 with effective batch size 8. The script reuses completed adapters and the validation ledger for each prefix. Adapter directories use `output/lora/qwen3-0.6b_v3_nXXXX_fp16_mixed`, with a suffix such as `_mb2` for a larger micro-batch. The corresponding validation results are under `output/lora_eval/`.

## BGE Small + logistic regression Student

Run the complete 13-checkpoint October validation experiment from the repository root on the Quadro RTX 6000.

```bash
uv run --locked --extra embedding-lr python code/student_embedding_lr.py --device cuda
```

The experiment freezes `BAAI/bge-small-en-v1.5` and caches the 384-dimensional complaint embeddings in `.cache/embedding_lr`. The n0 head uses the five v3 category descriptions as zero-shot prototypes. The remaining heads fit multinomial LR on the chronological DeepSeek-labeled history prefixes also used by LoRA, at 100, 200, 500 and each 1,000 through 9,000. Inference records all five probabilities per case. The confidence gate replays the archived DeepSeek predictions for low-confidence cases.

The [validation report](result/validation/bge-small-en-v1.5-v3-embedding-lr/note.md) links the per-checkpoint parameters, validation metrics, prediction ledgers and Student/Teacher comparison plots. Use `--output` for a separate result directory or `--batch-size` to adjust the embedding batch size.

## Confidence-gated Teacher fallback (offline)

[CGTF 实验报告](result/validation/gemma3-270m-it-v3-cgtf/note.md) evaluates confidence-gated Teacher fallback on all 12 archived Gemma validation checkpoints. A request is escalated only when `confidence < threshold`; equality retains the Student output. The primary mode preserves original strict outputs, including failures; a separate five-way mode provides a diagnostic comparison. No new Teacher API calls are made.

```bash
python3 -m unittest discover -s code -p 'test_confidence_gated_fallback.py'
MPLCONFIGDIR=/tmp/llm-jit-matplotlib python3 code/confidence_gated_fallback.py
```

The script writes threshold metrics, routing counts, per-request hybrid validation ledgers and plots under `result/validation/gemma3-270m-it-v3-cgtf/`. Matplotlib is required for plots; `--skip-plots` runs the standard-library scoring/export path. Add `--export-threshold 0.85` to export another threshold's detailed records. This follow-up experiment reports full-cohort hybrid quality separately from quality on retained Student cases; Teacher request counts are preserved for subsequent cost analysis.

## Code and checks

| Module | Responsibility |
|---|---|
| `code/mortgage_data.py` | Three-partition cohort audit, complete selections, clipping and manifests |
| `code/mortgage_prompt.py` | Shared labels, versioned prompts, messages and strict parsing |
| `code/teacher_client.py` | SiliconFlow Teacher profiles, model-specific request settings and isolated output paths |
| `code/teacher_baseline.py` / `code/teacher_collection.py` | Full-cohort collection, retries, concurrency, resume and exports |
| `code/student_baseline.py` | Student inference, resume, Teacher pairing and scoring |
| `code/student_lora.py` | History-only completion SFT, target token audit and PEFT adapter export |
| `code/student_embedding_lr.py` | BGE embeddings, LR checkpoints, validation probabilities and fallback replay |
| `code/evaluation.py` / `code/run_store.py` | Metrics, raw attempt ledger, locking and atomic exports |

Run offline checks without API calls or model downloads:

```bash
uv run --locked python -m unittest discover -s code -p 'test_*.py'
```

Validation result: [Gemma 3 270M IT, full October validation, v3/CUDA](result/validation/gemma3-270m-it-v3-cuda/note.md). The completed 1,013-case Student run has 26.75% accuracy, 0.1591 macro-F1 and 83.61% valid output; Teacher pairing was incomplete at export. The report includes frozen source snapshots and independently checked statistics.

Historical evidence: [Gemma 3 pilot](result/pre-run%20300%20sample%20test/gemma3-270m-it-v3/note.md) · [Qwen pilot](result/pre-run%20300%20sample%20test/qwen3.5-0.8b-v3/note.md). Their original metadata remains in archived output/report directories.
