import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import modal


# =========================================================
# Paths / configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
REMOTE_TEST_FILE = "/data/test.jsonl"

MODEL_NAME = "Qwen/Qwen3-4B"

# Pin model revision so future HF updates do not change results.
MODEL_REVISION = "1cfa9a7"

# Model will be downloaded into the Modal image during build.
MODEL_DIR = "/models/qwen3-4b"

MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 384
GPU_MEMORY_UTILIZATION = 0.88


# =========================================================
# OpenPII taxonomy
# =========================================================

ALLOWED_TYPES = frozenset({
    "DATE", "GIVENNAME", "SURNAME", "EMAIL", "CITY", "TITLE",
    "TELEPHONENUM", "AGE", "STREET", "BUILDINGNUM", "ZIPCODE",
    "IDCARDNUM", "CREDITCARDNUMBER", "DRIVERLICENSENUM", "GENDER",
    "TAXNUM", "SEX", "SOCIALNUM", "PASSPORTNUM",
})


# =========================================================
# Prompt
# =========================================================

SYSTEM_PROMPT = """
You are a PII extraction system.

Extract every personally identifiable information entity
from the input text.

Use ONLY these entity types:

DATE
GIVENNAME
SURNAME
EMAIL
CITY
TITLE
TELEPHONENUM
AGE
STREET
BUILDINGNUM
ZIPCODE
IDCARDNUM
CREDITCARDNUMBER
DRIVERLICENSENUM
GENDER
TAXNUM
SEX
SOCIALNUM
PASSPORTNUM

Rules:

1. Copy entity text exactly as it appears in the input.
2. Keep GIVENNAME and SURNAME separate.
3. Keep STREET and BUILDINGNUM separate.
4. Use only the entity types listed above.
5. Do not invent entities.
6. If no PII exists, return an empty entities list.
7. Return ONLY valid raw JSON.
8. Do not use Markdown.
9. Do not provide explanations.

Required format:

{"entities":[{"text":"exact substring","type":"TYPE"}]}
""".strip()


# =========================================================
# Modal
# =========================================================

app = modal.App("privacyguard-baseline")

# Reads HF_TOKEN from your local project .env.
# .env:
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxx

hf_secret = modal.Secret.from_dotenv(PROJECT_ROOT, filename=".env")


# =========================================================
# Model download during IMAGE BUILD
# =========================================================

def download_model(model_name: str, revision: str, target_dir: str):
    import os
    from huggingface_hub import snapshot_download
    print(f"Downloading {model_name} revision={revision}")
    snapshot_download(
        repo_id=model_name,
        revision=revision,
        local_dir=target_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Model saved to {target_dir}")


image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .run_function(
        download_model,
        args=(MODEL_NAME, MODEL_REVISION, MODEL_DIR),
        secrets=[hf_secret],

        # CPU build step.
        # No L4 is used for downloading the model.
        cpu=4,
        memory=16384,
        timeout=60 * 60,
    )
    .add_local_file(str(LOCAL_TEST_FILE), REMOTE_TEST_FILE)
)


# =========================================================
# JSON parsing
# =========================================================

def parse_response(raw_text: str):
    """
    Returns:

    entities
    valid_json
    recovered_json

    valid_json=True only when the original model response
    is directly valid JSON.

    Recovery exists only for debugging.
    """

    raw_text = raw_text.strip()

    # -----------------------------------------------------
    # Strict JSON
    # -----------------------------------------------------

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
            return parsed["entities"], True, False
    except (json.JSONDecodeError, TypeError):
        pass

    # -----------------------------------------------------
    # Markdown recovery
    # -----------------------------------------------------

    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
                return parsed["entities"], False, True
        except (json.JSONDecodeError, TypeError):
            pass

    # -----------------------------------------------------
    # Extract outer JSON
    # -----------------------------------------------------

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw_text[start : end + 1])
            if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
                return parsed["entities"], False, True
        except (json.JSONDecodeError, TypeError):
            pass
    return [], False, False


# =========================================================
# Span alignment
# =========================================================

def align_entities_to_text(source_text: str, predicted_entities: list):
    """
    Model predicts:

    {
        "text": "john@gmail.com",
        "type": "EMAIL"
    }

    Python determines start/end offsets.

    Only exact OpenPII labels are accepted.
    """

    aligned = []
    used_spans = set()
    for entity in predicted_entities:
        if not isinstance(entity, dict):
            continue
        value = entity.get("text")
        label = entity.get("type")
        if not isinstance(value, str):
            continue
        if not isinstance(label, str):
            continue
        value = value.strip()
        label = label.strip().upper()
        if not value:
            continue

        # Reject labels outside our
        # official OpenPII taxonomy.
        if label not in ALLOWED_TYPES:
            continue

        # -------------------------------------------------
        # Find exact text occurrence
        # -------------------------------------------------

        position = source_text.find(value)
        while position != -1:
            start = position
            end = position + len(value)
            key = start, end, label
            if key not in used_spans:
                used_spans.add(key)
                aligned.append({"text": value, "type": label, "start": start, "end": end})
                break
            position = source_text.find(value, position + 1)
    return aligned


# =========================================================
# Load evaluation samples
# =========================================================

def load_samples(path: str, limit: int):
    samples = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if len(samples) >= limit:
                break
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            missing = {"uid", "text", "entities"} - set(sample)
            if missing:
                raise ValueError(f"Missing fields on line {line_number}: {sorted(missing)}")
            samples.append(sample)
    if not samples:
        raise ValueError("No test samples loaded.")
    return samples


# =========================================================
# Remote GPU evaluation
# =========================================================

@app.function(image=image, gpu="L4", cpu=4, memory=24576, timeout=60 * 60)
def run_baseline(limit: int = 100):

    from vllm import LLM, SamplingParams
    if limit < 1:
        raise ValueError("limit must be >= 1")

    # -----------------------------------------------------
    # Load test samples
    # -----------------------------------------------------

    samples = load_samples(REMOTE_TEST_FILE, limit)
    print(f"\nLoaded {len(samples)} evaluation samples.")

    # -----------------------------------------------------
    # Initialize vLLM
    # -----------------------------------------------------

    print("\nLoading Qwen3-4B with vLLM...")
    load_start = time.perf_counter()
    llm = LLM(

        # Model already exists
        # inside the Modal image.
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,

        # Many requests share the
        # exact same system prompt.
        enable_prefix_caching=True,
        trust_remote_code=False,
    )
    model_load_seconds = time.perf_counter() - load_start
    print(f"Model ready in {model_load_seconds:.2f}s")

    # -----------------------------------------------------
    # Construct ALL conversations first
    # -----------------------------------------------------

    conversations = []
    for sample in samples:
        conversations.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Extract all PII from this text:\n\n" + sample["text"]},
        ])

    # -----------------------------------------------------
    # Deterministic decoding
    # -----------------------------------------------------

    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=42)

    # -----------------------------------------------------
    # vLLM batch inference
    # -----------------------------------------------------

    print("\nStarting vLLM batch inference...")
    generation_start = time.perf_counter()
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        use_tqdm=True,

        # Officially supported by
        # vLLM 0.21.0.
        #
        # Prevent Qwen3 from spending
        # tokens on reasoning.
        chat_template_kwargs={"enable_thinking": False},
    )
    generation_seconds = time.perf_counter() - generation_start

    # -----------------------------------------------------
    # Post-processing
    # -----------------------------------------------------

    results = []
    finish_reasons = Counter()
    total_output_tokens = 0
    recovered_count = 0
    for sample, output in zip(samples, outputs):
        completion = output.outputs[0]
        response = completion.text.strip()
        finish_reason = completion.finish_reason or "unknown"
        finish_reasons[finish_reason] += 1
        total_output_tokens += len(completion.token_ids)
        raw_entities, valid_json, recovered_json = parse_response(response)
        if recovered_json:
            recovered_count += 1
        predicted_entities = align_entities_to_text(sample["text"], raw_entities)
        results.append({
            "uid": sample["uid"],
            "gold_entities": sample["entities"],
            "predicted_entities": predicted_entities,
            "valid_json": valid_json,
            "recovered_json": recovered_json,
            "finish_reason": finish_reason,
            "raw_response": response,
        })

    # -----------------------------------------------------
    # Speed stats
    # -----------------------------------------------------

    samples_per_second = len(samples) / generation_seconds
    output_tokens_per_second = total_output_tokens / generation_seconds if generation_seconds else 0
    stats = {
        "sample_count": len(samples),
        "model_load_seconds": model_load_seconds,
        "generation_seconds": generation_seconds,
        "samples_per_second": samples_per_second,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second": output_tokens_per_second,
        "recovered_json_count": recovered_count,
        "finish_reasons": dict(finish_reasons),
    }
    print("\nRemote timing:")
    print(json.dumps(stats, indent=2))
    return {"results": results, "stats": stats}


# =========================================================
# Local entrypoint
# =========================================================

@app.local_entrypoint()
def main(limit: int = 100):

    total_start = time.perf_counter()
    payload = run_baseline.remote(limit)
    total_seconds = time.perf_counter() - total_start
    results = payload["results"]
    stats = payload["stats"]
    stats["total_round_trip_seconds"] = total_seconds

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    predictions_file = results_dir / "baseline_predictions.jsonl"
    with predictions_file.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    try:
        from evaluation.metrics import calculate_metrics
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT))
        from evaluation.metrics import calculate_metrics
    metrics = calculate_metrics(results)

    # -----------------------------------------------------
    # Save metrics + timing
    # -----------------------------------------------------

    metrics_file = results_dir / "baseline_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metrics": metrics,
                "timing": stats,
                "configuration": {
                    "model": MODEL_NAME,
                    "revision": MODEL_REVISION,
                    "gpu": "L4",
                    "dtype": "bfloat16",
                    "max_model_len": MAX_MODEL_LEN,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Console report
    # -----------------------------------------------------

    print("\n" + "=" * 56)
    print("              BASELINE EVALUATION")
    print("=" * 56)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:26s}: {value:.4f}")
        else:
            print(f"  {key:26s}: {value}")
    print("-" * 56)
    print(f"  generation_seconds        : {stats['generation_seconds']:.2f}")
    print(f"  samples_per_second        : {stats['samples_per_second']:.3f}")
    print(f"  output_tokens_per_second  : {stats['output_tokens_per_second']:.2f}")
    print(f"  model_load_seconds        : {stats['model_load_seconds']:.2f}")
    print(f"  round_trip_seconds        : {stats['total_round_trip_seconds']:.2f}")
    print(f"  recovered_json            : {stats['recovered_json_count']}")
    print(f"  finish_reasons            : {stats['finish_reasons']}")
    print("=" * 56)
    print(f"\nPredictions -> {predictions_file}")
    print(f"Metrics     -> {metrics_file}")