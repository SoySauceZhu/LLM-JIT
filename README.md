# LLM-JIT

[System and experiment design](https://docs.google.com/document/d/1LyIxZPZP4Ax1GHQ-LLyzqF2JipfCaxy5SB8kU0B53ug/edit?usp=sharing) · [Local experiment design](Experiment%20Design.md)

The primary experiment is **Gemma 4 31B IT Teacher → Gemma 3 270M IT Student**, using the frozen v3 prompt. SmolLM2-135M and Qwen3.5-0.8B remain exploratory pilot comparisons. Main-study commands explicitly select `--student-model gemma3-270m-it`; the CLI's legacy default remains SmolLM2 for compatibility. The [experiment design](Experiment%20Design.md) records the model-selection rationale, pilot evidence and planned Gemma 3 LoRA learning curves.

## Code layout

| Module | Responsibility |
|---|---|
| `code/teacher_client.py` | SiliconFlow Gemma client and request settings |
| `code/teacher_baseline.py` | Teacher preview, collection and export CLI |
| `code/teacher_collection.py` | API scheduling, retries, pause/resume and chronological history commits |
| `code/teacher_smoke.py` | One real API request on a development complaint |
| `code/student_baseline.py` | Unadapted SmolLM2/Qwen/Gemma inference, resume and paired evaluation |
| `code/mortgage_data.py` | Cohort audit, selection, frozen manifests and narrative clipping |
| `code/mortgage_prompt.py` | Shared labels, prompts, messages and strict parsing |
| `code/evaluation.py` | Shared classification metrics and Teacher agreement |
| `code/run_store.py` | SQLite ledger, run locking, history prefixes and atomic file exports |

The previous entry points `baseline_gemma.py`, `gemma_teacher.py`, `smoke_gemma.py` and `prerun_smollm2.py` have been renamed; use the commands below. Archived pilot outputs remain in their original directories.

## Environment and frozen configuration

Run from the repository root with Python 3.12. Use the project virtual environment and install the direct dependency versions used by the verified environment:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The virtual environment reuses existing system packages such as Torch while allowing project-local version pins. Activate it before running the commands below. Transformers 5.5.4 requires `huggingface-hub>=1.5.0,<2.0`; this project pins 1.11.0. A newer global Hugging Face CLI installation can bring an incompatible Hub 2.x, so use the project environment for inference. `.venv/` is git-ignored.

`requirements.txt` pins direct dependencies rather than a portable full lockfile. On the current machine, `pip check` flags the installed Torch wheel's platform tag; importing Torch and running CPU tensor operations both succeed. The offline tests also exercise the generation wrapper with CPU tensors and a simulated model.

Put your SiliconFlow international key in `.env` as `SILICONFLOW_API_KEY`, or set that environment variable. Credentials are needed only for pending Teacher calls. Preview, export and local Student inference do not require an API key.

Both models default to **v3** after the four prompt pilots. Use `--prompt-version v1`, `v2`, `v3` or `v4` to reproduce another version. Gemma uses `google/gemma-4-31B-it` at `https://api.siliconflow.com/v1`, temperature 0 and max_tokens 16. This is the provider's model identifier, not an immutable weights revision; the ledger records returned model identifiers, response IDs and timestamps. SmolLM2 uses `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, CPU/FP32, four threads, batch size 1, greedy decoding and at most 16 generated tokens per attempt.

Teacher and Student use the same prompt messages and visible narrative. Narratives over 3,000 tokens of the pinned Student tokenizer retain the first 2,000 and final 1,000. Model weights/tokenizer files are downloaded by Hugging Face only when needed by a real run. A preview does not load either.

The immutable source `data/mortgage_complaint.csv` contains 12,196 unique complaints, including the archived January development cases. Its frozen 2026-11-21 ordered-ID SHA-256 is `3c59aa321b65ea493b579cd503af0aeb580fa9c28ee4cabe4b15d136ab123dfa`. Loading audits counts, chronological order, unique IDs/text hashes, text integrity and temporal splits against this source snapshot.

The **formal experiment contains only 11,239 requests in three partitions**:

| Partition | Receipt window | Requests | Role |
|---|---|---:|---|
| `history` (training stream) | February–September | 8,202 | Ordered Teacher requests and training prefixes |
| `validation` | October | 1,013 | Future requests for baseline/adapter comparison and selection |
| `test` | November–December | 2,024 | Final held-out evaluation |

`mortgage_data.load_experiment()` returns these three partitions and excludes January. `load_development()` returns only the archived 957 January cases for explicit pilot reproduction or smoke checks. `load_cohort()` is the complete source-audit interface. The baseline CLIs select the appropriate loader: history/validation use formal data; only an explicit `--split development` uses January. The source CSV, fingerprints and existing run configurations retain their original values, so old pilot results remain reproducible. Final test scoring is reserved for the later evaluation stage and is not exposed by the current baseline CLIs.

## Full baselines

Without `--run`, either baseline CLI only previews the selected input and output path:

```bash
python3 code/teacher_baseline.py --split history
python3 code/teacher_baseline.py --split validation
python3 code/student_baseline.py --student-model gemma3-270m-it --split validation
```

Start the chronological history collection and the two validation baselines explicitly:

```bash
python3 code/teacher_baseline.py --split history --run
python3 code/teacher_baseline.py --split validation --concurrency 4 --run
python3 code/student_baseline.py --student-model gemma3-270m-it --device mps --split validation --run
```

The default split is `validation`. History and validation always use the complete split in frozen stream order; `--size` and `--seed` are accepted only for development. History requires concurrency 1 so that prefix membership follows arrival order. Validation concurrency defaults to 1 and can be increased explicitly.

Default output directories for v3:

```text
output/baseline_v3/teacher_history/
output/baseline_v3/teacher_validation/
output/baseline_v3/student_gemma3-270m-it_validation/
```

Use `--output PATH` to select another run directory. Rerun the same command to resume. Input hashes, prompt contents, model settings and environment are frozen in `run.sqlite`; changed configurations require another output directory. Validation Teacher concurrency may change on resume and is recorded for each attempt. Only one writer can operate on a run directory at a time.

A timeout or invalid Teacher reply gets one retry; other API errors, including HTTP 429, pause collection. Follow any recorded `Retry-After` value and resume with lower validation concurrency. On Ctrl+C, collection stops scheduling new work, drains in-flight API requests and exports committed results. A hard process termination during an uncommitted API call may cause a repeated charge when resumed. Student commits each generation attempt, so an interruption between two attempts preserves the remaining retry budget. Local runtime errors pause instead of becoming classification failures.

## Comparing Student models

`--student-model` selects `smollm2-135m` (default), `qwen3.5-0.8b`, or `gemma3-270m-it`. Qwen uses the post-trained `Qwen/Qwen3.5-0.8B` checkpoint at revision `2fc06364715b967f1860aea9cf38778875588b17`, its native tokenizer/chat template and explicit `enable_thinking=False`. All Students default to CPU/FP32, four threads, greedy decoding, a 16-token limit and the same strict A–E parser. Use `--device mps` for Apple Metal acceleration where available, still in FP32; use a separate `--output` directory when changing devices. Device changes can change predictions and timings, so the chosen device is frozen in each run configuration. The text-only backbone is loaded with `AutoModelForCausalLM`; the vision tower is not used. Qwen's hybrid attention falls back to native PyTorch on this machine; optional CUDA kernels are not installed.

The **clipping tokenizer stays fixed to SmolLM2 for all models**, preserving the exact narratives already seen by the Teacher. In `config.json`, `tokenizer`/`revision` identify this shared clipping tokenizer; `model_revision`, `inference_tokenizer` and `inference_tokenizer_revision` identify the selected Student. Token counts measured during inference use each model's native tokenizer.

Run Qwen on the same 300 January pilot complaints and pair with the archived v3 Teacher:

```bash
python3 code/student_baseline.py --student-model qwen3.5-0.8b --device mps --split development --size 300 --seed 42 --prompt-version v3 --teacher-run output/gemma_dev_n300_s42_v3 --output output/qwen35_dev_n300_s42_v3_mps --run
```

Omit `--output` to use `output/baseline_v3/student_qwen3.5-0.8b_development_n300_s42/`; full validation uses `student_qwen3.5-0.8b_validation/`. SmolLM2 retains its existing default directory. Model settings are frozen in the ledger, so another model cannot resume into the same run. Generation stops on both the native tokenizer end-of-turn ID and the model end-of-text ID. In this Qwen checkpoint those IDs differ; the terminal stop token is excluded from the parsed reply, while other generated special tokens remain visible to strict parsing. Qwen weights are cached under the git-ignored `.cache/huggingface/` directory. The first actual Qwen run requires downloading approximately 1.8 GB; previews and exports do not download weights. Existing matching Teacher labels can be reused without new API calls.

The completed v3 300-case Qwen pilot, paired predictions and limitations are recorded in [the pilot report](result/pre-run%20300%20sample%20test/qwen3.5-0.8b-v3/note.md).

### Gemma 3 270M IT

`gemma3-270m-it` uses the instruction-tuned `google/gemma-3-270m-it` checkpoint at revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`. Its native chat template folds the initial system instruction into the first user turn and maps assistant turns to `model`. The shared instructions, five demonstrations and complaint text are passed through unchanged; no custom template or Qwen thinking option is applied. Generation also stops on the native `<end_of_turn>` token, in addition to the tokenizer/model EOS IDs. Loading uses `AutoModelForCausalLM` and the installed Transformers version already supports `gemma3_text`.

Access to the [official model repository](https://huggingface.co/google/gemma-3-270m-it) requires reviewing the Gemma conditions and obtaining access with your Hugging Face account. Configure an authorized `HF_TOKEN` in the environment or run `hf auth login` locally before the first actual download. Credentials should stay local. Weights/tokenizer files use the git-ignored `.cache/huggingface/` cache; previews and offline fixture tests do not download them or require authentication. This integration has passed 32 offline tests and completed a real 300-case MPS/FP32 v3 pilot; see [the Gemma 3 pilot report](result/pre-run%20300%20sample%20test/gemma3-270m-it-v3/note.md) for results, paired predictions and limitations.

Preview the same 300-case development sample:

```bash
python3 code/student_baseline.py --student-model gemma3-270m-it --split development --size 300 --seed 42 --prompt-version v3
```

After model access is configured, run with the archived Teacher labels:

```bash
python3 code/student_baseline.py --student-model gemma3-270m-it --device mps --split development --size 300 --seed 42 --prompt-version v3 --teacher-run output/gemma_dev_n300_s42_v3 --output output/gemma3_270m_dev_n300_s42_v3_mps --run
```

The default directory is `output/baseline_v3/student_gemma3-270m-it_development_n300_s42/`; full validation uses `student_gemma3-270m-it_validation/`. The same resume, export and Teacher-pairing rules apply. Metal acceleration is available through `--device mps` in a GPU-enabled execution environment; choose a separate output directory when changing the device.

## Outputs, pairing and export

Every run writes:

- `run.sqlite`: frozen configuration and individual attempts, raw replies, labels, token usage, timings and statuses.
- `config.json`: a readable configuration and ordered input manifest, without a second narrative copy.
- `predictions.csv`: one row per selected complaint in frozen order, including pending cases.
- `metrics.json`: completion status, terminal-case scores, five-class per-class metrics, confusion matrix with a failure column, and known usage across all attempts.

Only terminal cases enter classification scores. An exhausted failure is incorrect; pending cases are not classified as failures. Partial reports include `complete=false`, `provisional=true`, `n_expected`, `n_terminal`, `n_pending` and `metric_denominator=terminal_cases`. Empty runs have null accuracy/F1. Final reports include all selected cases. Currency results remain `unpriced`; no API or hardware rates are guessed.

Student automatically looks for the same-version Teacher run at its default path. Alternatively, pass `--teacher-run PATH`, including a compatible archived pilot directory. It checks ordered IDs, reference labels, clipped-input hashes, clipping metadata, tokenizer settings and exact prompt content before comparing labels. Missing/incomplete Teacher records leave agreement null; an explicitly missing path or mismatched configuration is an error. Full Teacher agreement is published only when both runs are complete, using all selected cases as the denominator and requiring matching legal labels. CFPB scores and Teacher agreement measure different things.

The first-token A–E choice remains a separate diagnostic and never replaces a failed free-generation reply. Student output retains first/final replies and all attempts in the ledger.

Regenerate reports from committed records, including after the Teacher finishes, without new inference:

```bash
python3 code/teacher_baseline.py --export-only --output output/baseline_v3/teacher_history
python3 code/student_baseline.py --export-only --output output/baseline_v3/student_gemma3-270m-it_validation
python3 code/student_baseline.py --export-only --output output/baseline_v3/student_gemma3-270m-it_validation --teacher-run output/baseline_v3/teacher_validation
```

With `--export-only --output`, the stored configuration controls the export. The destination files are staged and each replacement is atomic. New CLI runs do not overwrite or migrate archived pilot ledgers; pairing reads those archives in SQLite read-only mode.

## History labels for later training

Each final history outcome commits the attempt, replay cursor, any accepted label and any newly reached prefix in one transaction. Only a valid response becomes a training label, regardless of agreement with CFPB. Failure cases advance the arrival cursor without advancing the success index.

History exports additionally contain:

- `history_labels.csv`: success index, arrival index, complaint ID, accepted attempt ID, Teacher label, input hash, original sequence number and receipt timestamp. It contains no CFPB reference label or narrative.
- `prefixes/n100.json`, `n200.json`, `n400.json`, `n800.json`, `n1600.json`, `n3200.json`, `n6400.json`: frozen cumulative members, trigger arrival and membership hash. The hash is SHA-256 of `json.dumps(members, sort_keys=True)` encoded as UTF-8.
- `metrics.json` lists every milestone, including those not reached yet.

Each prefix includes the earlier prefixes. A later trainer can join IDs to the original inputs and verify clipping/input hashes; it must consume these Teacher labels, not CFPB references. This stage does not train adapters or launch training jobs.

## Development pilots and checks

Reproduce the January stratified sample (default size 300, seed 42) in new run directories:

```bash
python3 code/teacher_baseline.py --split development --size 300 --seed 42 --prompt-version v3 --run --concurrency 4
python3 code/student_baseline.py --student-model gemma3-270m-it --device mps --split development --size 300 --seed 42 --prompt-version v3 --run
```

Their default paths are `output/baseline_v3/teacher_development_n300_s42/` and `student_gemma3-270m-it_development_n300_s42/`. The seed-42 membership matches the archived 300-case pilot. To pair with an archive directly, supply the matching `--teacher-run output/gemma_dev_n300_s42_v3` to the Student command. The original SmolLM2 directories remain available for explicit `--student-model smollm2-135m` runs.

A real single-call connectivity check, using shared clipping and the selected prompt:

```bash
python3 code/teacher_smoke.py --prompt-version v3
```

Run offline regression and acceptance tests (synthetic model replies; no API calls or model downloads):

```bash
python3 -m unittest discover -s code -p 'test_*.py'
```

LoRA training, TF–IDF baselines, final test scoring and deployment cost replay are subsequent stages.
