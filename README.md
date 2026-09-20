# PrivacyGuard

PrivacyGuard evaluates language models on personally identifiable information
(PII) extraction and redaction tasks.

## Setup

Install [uv](https://docs.astral.sh/uv/) and use Python 3.12 or newer:

```powershell
uv sync
```

Inspect the source dataset with:

```powershell
uv run python src/data/inspect_dataset.py
```

Place the evaluation split at `data/processed/test.jsonl`. Each non-empty line
must be a JSON object containing `uid`, `text`, and `entities` fields.

## Modal baseline

Authenticate with Modal once, then run the baseline remotely:

```powershell
uv run modal setup
uv run modal run evaluation/baseline_modal.py
```

Predictions are written to
`results/baseline_predictions.jsonl`. The GPU image installs its own runtime
packages (`torch`, `transformers`, `accelerate`, and `bitsandbytes`), so those
large CUDA dependencies are not required in the local environment.
