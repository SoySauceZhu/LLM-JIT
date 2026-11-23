# LLM-JIT

## System Desgin & Experiment Design
[See this google doc](https://docs.google.com/document/d/1LyIxZPZP4Ax1GHQ-LLyzqF2JipfCaxy5SB8kU0B53ug/edit?usp=sharing)

## Gemma Teacher pilot

From the repository root, set `SILICONFLOW_API_KEY` in `.env` to your SiliconFlow international account key. Preview the January sample, then start or resume the run.

```bash
python3 -m pip install openai python-dotenv huggingface-hub tokenizers
python3 code/baseline_gemma.py
python3 code/baseline_gemma.py --run
```

To run up to four requests at once, use `python3 code/baseline_gemma.py --run --concurrency 4`. The default is one. You can resume a partially completed serial run with a different concurrency in the same output directory. If the earlier script is still running, let it finish or stop it first.

The default is 300 development cases, sampled proportionally by CFPB reference label with seed 42. Use `--size` and `--seed` to change the sample. Gemma and SmolLM2 use [the same v3 prompt](code/prompts/mortgage_v3.toml). It includes expanded category boundaries plus five synthetic complaint/answer pairs, one per class, followed by the actual complaint and a request for one letter. The first run downloads the pinned SmolLM2 tokenizer. For narratives over 3,000 tokens it preserves the first 2,000 and last 1,000.

The default Teacher run directory is `output/gemma_dev_n300_s42_v3/`. Its `run.sqlite` stores the exact prompt and examples, ordered sample, decoding settings and individual attempts. A timeout or an invalid reply gets one retry. Other API errors pause collection. Rerun the same command to resume from committed attempts. An interruption while an API request is in flight can lead to a repeated charge for that request.

Run `python3 code/prerun_smollm2.py` to evaluate the unadapted Student with v3 on the same frozen complaints. It reports CFPB scores immediately and Teacher agreement when a completed matching v3 Teacher run is available at the default path. Both pilot scripts accept `--prompt-version v1` or `--prompt-version v2` to reproduce the archived prompt versions.

After collection, `predictions.csv` pairs each reference label with its Teacher output by complaint ID. `metrics.json` reports accuracy, five-class macro-F1, per-class recall and false-positive rate, a confusion matrix, valid-output rate, and known token usage across attempts. A final failure counts as incorrect. These scores measure agreement with the CFPB reference labels on the January pilot.

Parallel responses are saved as they arrive. The CSV follows the original sample order. On a rate limit (HTTP 429) or Ctrl+C, the runner stops taking new cases and saves responses from requests already in flight. After a rate limit, respect the provider's `Retry-After` value when supplied and resume with a lower concurrency. The ledger records the configured concurrency per attempt, including when you resume with a different setting.
