# Do safety warnings survive multi-agent LLM handoffs?

Code, cases and per-case logs for the study **"Where Safety Warnings Get Lost in Multi-Agent LLM Pipelines: A Small-Model Study on Drug-Label Cases"** (Mohammed Saqlain, 2026).

The study gives a team of LLM agents a default drug order and text from the real openFDA label of that drug. It then asks whether a contraindication seen by the first agent still changes the final order after a documentation handoff and an optional review round. It compares a three-agent pipeline, the same pipeline with a review round, and one agent that sees the chart. Two small local models were run on all 1,000 cases: Gemma 4 E2B and Qwen2.5 3B (both Q4_K_M).

> This is a research benchmark on label text. It is not medical advice and it does not show that any system is safe or unsafe in a clinic.

## Layout

```
data/
  cases.json             the 1,000 cases (explicit 459, infer 341, control 200)
  manifest.json          counts and the openFDA retrieval time (2026-10-07)
results/
  *.jsonl                one row per case per model, with the raw output of every role
  summary.json           output of src/analyze.py (tests H1, H2, H3)
  extra_stats.json       output of src/extra_analysis.py (post hoc analyses)
src/
  build_dataset.py       builds cases.json from the openFDA label API
  cases.py               loads data/cases.json
  run_experiment.py      runs the six roles per case against a local llama.cpp server
  analyze.py             main analysis (H1, H2, H3)
  extra_analysis.py      post hoc analyses
```

## Release

**v1.0** is the version cited in the paper.

## Reproduce the analysis (no model needed)

Python 3.12. The scripts use only the standard library.

```
python src/analyze.py          # writes results/summary.json
python src/extra_analysis.py   # writes results/extra_stats.json
```

## Re-run the models (optional, slow)

The full run took about 12.4 h (Gemma) and 36.9 h (Qwen) on a laptop CPU (Intel i5-8250U, no GPU offload).

`src/run_experiment.py` expects two local services:

1. `llama-server` (llama.cpp, build 11344, commit ec7630a64) on `127.0.0.1:8792`, started with `-t 8 -c 4096 -ngl 0 --reasoning off`. Requests use temperature 0 and seed 0.
2. A small model-switching wrapper on `127.0.0.1:8791` with `GET /api/status` (returns `{"phase": "ready", "model": "<gguf file name>"}`) and `POST /api/model` (body `{"name": "<gguf file name>"}`) that restarts `llama-server` with the requested model. The wrapper is not included here, because it is a separate local tool.

Set `LLM_MODELS_DIR` to the folder with the `.gguf` files. The script resumes from `results/<model>.jsonl`, so an interrupted run can continue.

The script was edited after the run only to remove a local file path and to make the Windows sleep-lock call a no-op on other systems. The experiment logic is unchanged.

To rebuild the cases, run `src/build_dataset.py`. The openFDA API changes over time, so use the released `data/cases.json` to match the reported results.

## Data

The label text comes from the public openFDA drug-label API (https://open.fda.gov). See the openFDA terms and license: https://open.fda.gov/terms/ and https://open.fda.gov/license/. openFDA states that its data must not be used to make decisions about medical care.

## License

Code: MIT (see `LICENSE`).
