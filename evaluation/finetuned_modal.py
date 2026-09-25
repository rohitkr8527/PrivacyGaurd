"""Fine-tuned Qwen3-4B evaluation with LoRA adapter on Modal GPUs."""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import modal


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
REMOTE_TEST_FILE = "/data/test.jsonl"

MODEL_NAME = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7"  # Pin revision for reproducibility
MODEL_DIR = "/models/qwen3-4b"

ADAPTER_REPO = "rohitkmr8527/privacyguard-qwen3-4b-qlora"
ADAPTER_DIR = "/models/privacyguard-qwen3-4b-qlora"
LORA_RANK = 16

MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 384
GPU_MEMORY_UTILIZATION = 0.88

ALLOWED_TYPES = frozenset({
    "DATE", "GIVENNAME", "SURNAME", "EMAIL", "CITY", "TITLE", "TELEPHONENUM",
    "AGE", "STREET", "BUILDINGNUM", "ZIPCODE", "IDCARDNUM", "CREDITCARDNUMBER",
    "DRIVERLICENSENUM", "GENDER", "TAXNUM", "SEX", "SOCIALNUM", "PASSPORTNUM",
})


# =========================================================
# Prompt
# =========================================================

SYSTEM_PROMPT = """
You are a PII extraction system.

Extract every personally identifiable information entity from the input text.

Use ONLY these entity types:
DATE, GIVENNAME, SURNAME, EMAIL, CITY, TITLE, TELEPHONENUM, AGE, STREET,
BUILDINGNUM, ZIPCODE, IDCARDNUM, CREDITCARDNUMBER, DRIVERLICENSENUM, GENDER,
TAXNUM, SEX, SOCIALNUM, PASSPORTNUM

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
# Modal App & Secret
# =========================================================

app = modal.App("privacyguard-finetuned-eval")
hf_secret = modal.Secret.from_dotenv(PROJECT_ROOT, filename=".env")


# =========================================================
# Asset Download (runs during image build)
# =========================================================

def download_assets(
    model_name: str, model_revision: str, model_dir: str,
    adapter_repo: str, adapter_dir: str,
):
    """Download base model and LoRA adapter from HuggingFace."""
    import os
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing.")

    # Validate adapter repo first (fail fast before downloading large model)
    api = HfApi(token=token)
    adapter_info = api.model_info(repo_id=adapter_repo, token=token)
    adapter_revision = adapter_info.sha
    if not adapter_revision:
        raise RuntimeError(f"Could not resolve revision for {adapter_repo}.")

    print(f"Resolved adapter {adapter_repo} revision={adapter_revision}")
    print(f"Downloading base model {model_name} revision={model_revision}")

    snapshot_download(
        repo_id=model_name, revision=model_revision,
        local_dir=model_dir, token=token,
    )

    print(f"Downloading LoRA adapter {adapter_repo} revision={adapter_revision}")

    snapshot_download(
        repo_id=adapter_repo, revision=adapter_revision,
        local_dir=adapter_dir, token=token,
    )

    # Validate adapter files
    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "adapter_config.json"
    weights_ok = (
        (adapter_path / "adapter_model.safetensors").is_file() or
        (adapter_path / "adapter_model.bin").is_file()
    )

    if not config_path.is_file():
        raise RuntimeError("adapter_config.json is missing from adapter repository.")
    if not weights_ok:
        raise RuntimeError("LoRA adapter weights are missing.")

    # Validate base model match
    with config_path.open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    adapter_base = adapter_config.get("base_model_name_or_path")
    accepted_base_names = {model_name, model_dir, "/models/qwen3-4b"}
    if adapter_base not in accepted_base_names:
        raise RuntimeError(
            f"Adapter/base-model mismatch. Expected {model_name!r}, found {adapter_base!r}."
        )

    # Normalize base model path in config
    if adapter_base != model_name:
        adapter_config["base_model_name_or_path"] = model_name
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2, ensure_ascii=False)

    # Validate LoRA rank
    rank = adapter_config.get("r")
    if rank != LORA_RANK:
        raise RuntimeError(f"Unexpected LoRA rank. Expected {LORA_RANK}, found {rank!r}.")

    (adapter_path / "_resolved_revision.txt").write_text(adapter_revision, encoding="utf-8")
    print(f"Base model saved to {model_dir}")
    print(f"Adapter saved to {adapter_dir}")


# =========================================================
# Modal Image
# =========================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    .run_function(
        download_assets,
        args=(MODEL_NAME, MODEL_REVISION, MODEL_DIR, ADAPTER_REPO, ADAPTER_DIR),
        secrets=[hf_secret],
        cpu=4,
        memory=16384,
        timeout=60 * 60,
    )
    .add_local_file(str(LOCAL_TEST_FILE), REMOTE_TEST_FILE)
)


# =========================================================
# JSON Parsing Utilities
# =========================================================

def parse_response(raw_text: str):
    """Parse model response, attempting strict JSON first, then recovery.

    Returns: (entities, valid_json, recovered_json)
    """
    raw_text = raw_text.strip()

    # Strict JSON parse
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
            return parsed["entities"], True, False
    except (json.JSONDecodeError, TypeError):
        pass

    # Markdown code block recovery
    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
                return parsed["entities"], False, True
        except (json.JSONDecodeError, TypeError):
            pass

    # Extract outermost JSON object
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw_text[start : end + 1])
            if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
                return parsed["entities"], False, True
        except (json.JSONDecodeError, TypeError):
            pass

    return [], False, False


def align_entities_to_text(source_text: str, predicted_entities: list):
    """Align predicted entities to source text spans.

    Model predicts: {"text": "john@gmail.com", "type": "EMAIL"}
    We compute start/end offsets via exact substring match.
    """
    aligned = []
    used_spans = set()

    for entity in predicted_entities:
        if not isinstance(entity, dict):
            continue

        value = entity.get("text")
        label = entity.get("type")
        if not isinstance(value, str) or not isinstance(label, str):
            continue

        value = value.strip()
        label = label.strip().upper()
        if not value or label not in ALLOWED_TYPES:
            continue

        # Find exact text occurrence
        position = source_text.find(value)
        while position != -1:
            start = position
            end = position + len(value)
            key = (start, end, label)

            if key not in used_spans:
                used_spans.add(key)
                aligned.append({"text": value, "type": label, "start": start, "end": end})
                break

            position = source_text.find(value, position + 1)

    return aligned


# =========================================================
# Data Loading
# =========================================================

def load_samples(path: str, limit: int):
    """Load evaluation samples from JSONL file."""
    samples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
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
# Remote GPU Evaluation
# =========================================================

@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24576,
    timeout=60 * 60,
)
def run_finetuned(limit: int = 100):
    """Run evaluation on Modal GPU with vLLM + LoRA."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if limit < 1:
        raise ValueError("limit must be >= 1")

    # Load test samples
    samples = load_samples(REMOTE_TEST_FILE, limit)
    print(f"\nLoaded {len(samples)} evaluation samples.")
    if len(samples) != limit:
        raise RuntimeError(f"Requested {limit} samples but only {len(samples)} were loaded.")

    # Initialize vLLM
    print("\nLoading Qwen3-4B with vLLM...")
    load_start = time.perf_counter()

    llm = LLM(
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=True,  # System prompt is shared across requests
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_loras=1,
    )

    model_load_seconds = time.perf_counter() - load_start
    print(f"Model ready in {model_load_seconds:.2f}s")

    # Load adapter revision
    adapter_revision = (
        Path(ADAPTER_DIR) / "_resolved_revision.txt"
    ).read_text(encoding="utf-8").strip()

    lora_request = LoRARequest("privacyguard-pii", 1, ADAPTER_DIR)
    print(f"Using adapter {ADAPTER_REPO} revision={adapter_revision}")

    # Build conversations
    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all PII from this text:\n\n{sample['text']}"},
        ]
        for sample in samples
    ]

    # Sampling params (deterministic)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
    )

    # Batch inference
    print("\nStarting vLLM batch inference...")
    generation_start = time.perf_counter()

    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        lora_request=lora_request,
        use_tqdm=True,
        chat_template_kwargs={"enable_thinking": False},  # Prevent Qwen3 reasoning tokens
    )

    generation_seconds = time.perf_counter() - generation_start

    # Post-process results
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

    # Compute stats
    samples_per_second = len(samples) / generation_seconds
    output_tokens_per_second = (
        total_output_tokens / generation_seconds if generation_seconds else 0
    )

    stats = {
        "sample_count": len(samples),
        "adapter_revision": adapter_revision,
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
# Local Entrypoint
# =========================================================

@app.local_entrypoint()
def main(limit: int = 100):
    """Local entrypoint to trigger remote evaluation."""
    total_start = time.perf_counter()
    payload = run_finetuned.remote(limit)
    total_seconds = time.perf_counter() - total_start

    results = payload["results"]
    stats = payload["stats"]
    stats["total_round_trip_seconds"] = total_seconds

    # Prepare output directory
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_file = results_dir / "finetuned_predictions.jsonl"
    with predictions_file.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Calculate metrics
    try:
        from evaluation.metrics import calculate_metrics
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT))
        from evaluation.metrics import calculate_metrics

    metrics = calculate_metrics(results)

    # Save metrics + timing
    metrics_file = results_dir / "finetuned_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "timing": stats,
            "configuration": {
                "model": MODEL_NAME,
                "revision": MODEL_REVISION,
                "adapter_repo": ADAPTER_REPO,
                "adapter_revision": stats["adapter_revision"],
                "lora_rank": LORA_RANK,
                "gpu": "L4",
                "dtype": "bfloat16",
                "max_model_len": MAX_MODEL_LEN,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        }, f, indent=2, ensure_ascii=False)

    # Console report
    print("\n" + "=" * 56)
    print("              FINE-TUNED EVALUATION")
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
