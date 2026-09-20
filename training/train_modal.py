import hashlib
import inspect
import json
import math
import os
import re
import shutil
import statistics
import time
import uuid
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
LOCAL_VAL   = PROJECT_ROOT / "data" / "processed" / "validation.jsonl"

REMOTE_TRAIN = "/data/train.jsonl"
REMOTE_VAL   = "/data/validation.jsonl"


# ---------------------------------------------------------------------------
# Model / training configuration
# ---------------------------------------------------------------------------

MODEL_NAME     = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7"
MODEL_DIR      = "/models/qwen3-4b"

WANDB_PROJECT = "privacyguard-qwen3-qlora"

MAX_LENGTH = 2048
SEED       = 42

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

LEARNING_RATE          = 2e-4
TARGET_EFFECTIVE_BATCH = 8
FULL_EPOCHS            = 1

# Proven configuration from the smoke benchmark.
# Switch to True and resume from the latest durable checkpoint on real OOM;
# this does not change the training objective.
USE_GRADIENT_CHECKPOINTING = False

CHECKPOINT_EVERY_STEPS = 100
SAVE_TOTAL_LIMIT       = 3


# ---------------------------------------------------------------------------
# Durable run namespace
#
# IMPORTANT: v6(This is 6th version of the training script) uses a NEW checkpoint directory so it cannot accidentally
# resume from a partial/incompatible checkpoint produced by an older script.
# ---------------------------------------------------------------------------

CHECKPOINT_VOLUME_NAME = "privacyguard-training-checkpoints"
CHECKPOINT_MOUNT       = "/checkpoints"
RUN_TAG                = "qwen3-4b-openpii25k-e1-r16-a32-bs8-seed42-v6"
RUN_DIR                = Path(CHECKPOINT_MOUNT) / RUN_TAG
FINAL_ADAPTER_DIR      = RUN_DIR / "final_adapter"
RUN_MANIFEST_PATH      = RUN_DIR / "run_manifest.json"
TRAINING_SUMMARY_PATH  = RUN_DIR / "training_summary.json"
COMPLETED_PATH         = RUN_DIR / "completed.json"
WANDB_RUN_ID_PATH      = RUN_DIR / "wandb_run_id.txt"


# Supported PII entity types — any other type is rejected during validation.
ALLOWED_TYPES = {
    "DATE", "GIVENNAME", "SURNAME", "EMAIL", "CITY", "TITLE",
    "TELEPHONENUM", "AGE", "STREET", "BUILDINGNUM", "ZIPCODE",
    "IDCARDNUM", "CREDITCARDNUMBER", "DRIVERLICENSENUM",
    "GENDER", "TAXNUM", "SEX", "SOCIALNUM", "PASSPORTNUM",
}

SYSTEM_PROMPT = """You are a PII extraction system.

Extract every personally identifiable information entity from the input text.

Use ONLY these entity types:
DATE, GIVENNAME, SURNAME, EMAIL, CITY, TITLE, TELEPHONENUM, AGE, STREET,
BUILDINGNUM, ZIPCODE, IDCARDNUM, CREDITCARDNUMBER, DRIVERLICENSENUM,
GENDER, TAXNUM, SEX, SOCIALNUM, PASSPORTNUM.

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
{\"entities\":[{\"text\":\"exact substring\",\"type\":\"TYPE\"}]}"""


# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App("privacyguard-qlora-training")

secrets = modal.Secret.from_dotenv(PROJECT_ROOT, filename=".env")

checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True
)


# ---------------------------------------------------------------------------
# Base image — bakes in pinned deps and downloads the base model at build time
# ---------------------------------------------------------------------------

def download_model(model_name: str, revision: str, target_dir: str):
    """Run during image build to cache the model weights in the image layer."""
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing")
    snapshot_download(repo_id=model_name, revision=revision,
                      local_dir=target_dir, token=token)


# These exact versions already built and completed the smoke training.
# huggingface_hub is deliberately not pinned; uv resolves the version
# compatible with Transformers 5.17.
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "torch==2.11.0",
        "transformers==5.17.0",
        "trl==1.13.0",
        "peft==0.21.0",
        "bitsandbytes==0.50.2",
        "accelerate==1.15.0",
        "datasets==5.0.1",
        "wandb==0.30.0",
    )
    .env({
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_PROJECT": WANDB_PROJECT,
    })
    .run_function(
        download_model,
        args=(MODEL_NAME, MODEL_REVISION, MODEL_DIR),
        secrets=[secrets],
        cpu=4, memory=16384, timeout=3600,
    )
    .add_local_file(str(LOCAL_TRAIN), REMOTE_TRAIN)
    .add_local_file(str(LOCAL_VAL), REMOTE_VAL)
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str):
    """Parse a .jsonl file and return a list of row dicts."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}"
                ) from exc
            missing = {"uid", "text", "entities"} - set(row)
            if missing:
                raise ValueError(
                    f"{path} line {line_number} missing fields: {sorted(missing)}"
                )
            rows.append(row)
    return rows


def validate_rows(rows, name: str):
    """Validate the processed training format before allocating an L4."""
    seen_uids: set = set()
    for index, row in enumerate(rows):
        uid      = str(row.get("uid", ""))
        text     = row.get("text")
        entities = row.get("entities")

        if not uid:
            raise ValueError(f"{name}[{index}] has an empty uid")
        if uid in seen_uids:
            raise ValueError(f"Duplicate uid in {name}: {uid}")
        seen_uids.add(uid)

        if not isinstance(text, str):
            raise TypeError(f"{name}[{index}] text is not a string")
        if not isinstance(entities, list):
            raise TypeError(f"{name}[{index}] entities is not a list")

        for ei, entity in enumerate(entities):
            missing = {"text", "type", "start", "end"} - set(entity)
            if missing:
                raise ValueError(
                    f"{name}[{index}].entities[{ei}] missing {sorted(missing)}"
                )
            entity_text = entity["text"]
            entity_type = entity["type"]
            start, end  = entity["start"], entity["end"]

            if entity_type not in ALLOWED_TYPES:
                raise ValueError(
                    f"Unsupported entity type {entity_type!r} in {name}[{index}]"
                )
            if not isinstance(start, int) or not isinstance(end, int):
                raise TypeError(
                    f"Non-integer span in {name}[{index}].entities[{ei}]"
                )
            if not (0 <= start < end <= len(text)):
                raise ValueError(
                    f"Invalid span ({start}, {end}) in {name}[{index}]"
                )
            if text[start:end] != entity_text:
                raise ValueError(
                    f"Span mismatch in {name}[{index}] uid={uid}: "
                    f"expected {entity_text!r}, got {text[start:end]!r}"
                )


def render_example(row, tokenizer):
    """Format a data row into (prompt, completion) strings for SFT."""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Extract all PII from this text:\n\n{row['text']}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    answer     = {"entities": [{"text": e["text"], "type": e["type"]} for e in row["entities"]]}
    completion = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    if tokenizer.eos_token:
        completion += tokenizer.eos_token
    return prompt, completion


def build_dataset(rows, tokenizer):
    """Convert a list of row dicts into a HuggingFace Dataset."""
    from datasets import Dataset
    records = [
        {"prompt": p, "completion": c}
        for p, c in (render_example(r, tokenizer) for r in rows)
    ]
    return Dataset.from_list(records)


def length_stats(rows, tokenizer, name: str):
    """Compute and print token-length statistics for a split."""
    if not rows:
        raise ValueError(f"{name} is empty")

    lengths = []
    for row in rows:
        prompt, completion = render_example(row, tokenizer)
        ids = tokenizer(prompt + completion, add_special_tokens=False,
                        truncation=False)["input_ids"]
        lengths.append(len(ids))

    ordered = sorted(lengths)

    def percentile(p: float):
        idx = int(round((len(ordered) - 1) * p))
        return ordered[max(0, min(idx, len(ordered) - 1))]

    stats = {
        "count":      len(lengths),
        "median":     int(statistics.median(lengths)),
        "p95":        percentile(0.95),
        "p99":        percentile(0.99),
        "max":        max(lengths),
        "over_limit": sum(l > MAX_LENGTH for l in lengths),
    }
    print(f"\n{name} token lengths")
    for key, value in stats.items():
        print(f"  {key:12s}: {value}")
    return stats


# ---------------------------------------------------------------------------
# TRL / Transformers compatibility helpers
# ---------------------------------------------------------------------------

def supported_params(callable_obj):
    """Return the set of parameter names accepted by callable_obj."""
    return set(inspect.signature(callable_obj).parameters)


def resolve_grad_accum(micro_batch_size: int):
    """Derive gradient accumulation steps from micro batch size."""
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1")
    if TARGET_EFFECTIVE_BATCH % micro_batch_size != 0:
        raise ValueError(
            f"micro_batch_size must divide TARGET_EFFECTIVE_BATCH="
            f"{TARGET_EFFECTIVE_BATCH}. Use 1, 2, 4, or 8."
        )
    return TARGET_EFFECTIVE_BATCH // micro_batch_size


def make_sft_kwargs(
    SFTConfig,
    *,
    smoke_test: bool,
    train_count: int,
    micro_batch_size: int,
    grad_accum: int,
    cpu_validation: bool,
):
    """Build and validate an SFTConfig kwargs dict, filtering to the installed API."""
    supported = supported_params(SFTConfig)

    # Smoke test uses a short fixed schedule; full run derives steps from data size.
    if smoke_test:
        max_steps, epochs, total_steps = 20, 1, 20
        eval_steps, logging_steps      = 10, 1
        save_strategy                  = "no"
        output_dir                     = "/tmp/privacyguard-smoke-v6"
    else:
        max_steps  = -1
        epochs     = FULL_EPOCHS
        steps_per_epoch = math.ceil(train_count / (micro_batch_size * grad_accum))
        total_steps     = steps_per_epoch * FULL_EPOCHS
        eval_steps, logging_steps = 1000, 10
        save_strategy             = "steps"
        output_dir                = str(RUN_DIR)

    warmup_steps = max(1, int(round(total_steps * 0.03)))

    requested = {
        "output_dir":                   output_dir,
        "overwrite_output_dir":         False,
        "num_train_epochs":             epochs,
        "max_steps":                    max_steps,
        "max_length":                   MAX_LENGTH,
        "completion_only_loss":         True,
        "packing":                      False,
        "loss_type":                    "nll",
        "per_device_train_batch_size":  micro_batch_size,
        "per_device_eval_batch_size":   1,
        "gradient_accumulation_steps":  grad_accum,
        "gradient_checkpointing":       USE_GRADIENT_CHECKPOINTING,
        "optim":                        "paged_adamw_8bit",
        "bf16":                         True,
        "fp16":                         False,
        "learning_rate":                LEARNING_RATE,
        "lr_scheduler_type":            "cosine",
        "warmup_steps":                 warmup_steps,
        "weight_decay":                 0.01,
        "max_grad_norm":                0.3,
        "eval_steps":                   eval_steps,
        "logging_strategy":             "steps",
        "logging_steps":                logging_steps,
        "logging_first_step":           True,
        "report_to":                    ["wandb"],
        "run_name":                     "privacyguard-smoke-v6" if smoke_test else RUN_TAG,
        "save_strategy":                save_strategy,
        "save_steps":                   CHECKPOINT_EVERY_STEPS,
        "save_total_limit":             SAVE_TOTAL_LIMIT,
        "save_safetensors":             True,
        # Must remain False so optimizer/scheduler state is available when
        # resuming after a Modal preemption.
        "save_only_model":              False,
        "ignore_data_skip":             False,
        "seed":                         SEED,
        "data_seed":                    SEED,
        "dataloader_num_workers":       0,
    }

    # Handle eval_strategy rename across TRL versions.
    if "eval_strategy" in supported:
        requested["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        requested["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError(
            "Installed SFTConfig supports neither eval_strategy nor evaluation_strategy"
        )

    # These args are non-negotiable — abort early if any are missing.
    critical = {
        "output_dir", "max_length", "completion_only_loss", "loss_type",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "gradient_checkpointing", "optim", "learning_rate", "warmup_steps",
        "bf16", "eval_steps", "report_to", "save_strategy", "save_steps",
    }
    missing = sorted(critical - supported)
    if missing:
        raise RuntimeError(
            "Installed TRL is incompatible. Unsupported critical SFTConfig args: "
            + ", ".join(missing)
        )

    # Drop optional args the installed version does not recognise.
    filtered = {k: v for k, v in requested.items() if k in supported}
    skipped  = sorted(set(requested) - set(filtered))
    if skipped:
        print("Optional SFTConfig args not supported and skipped: " + ", ".join(skipped))

    # Validate constructor compatibility on CPU without pretending CPU supports
    # the real BF16/bitsandbytes GPU configuration.
    if cpu_validation:
        validation_kwargs = dict(filtered)
        for key, override in (
            ("use_cpu", True), ("bf16", False), ("fp16", False),
            ("optim", "adamw_torch"), ("report_to", "none"),
        ):
            if key in supported:
                validation_kwargs[key] = override
        SFTConfig(**validation_kwargs)
        print("\nSFTConfig constructor validated with CPU-safe overrides.")
    else:
        # On the L4 this validates the exact real configuration BEFORE loading Qwen3-4B.
        SFTConfig(**filtered)

    return filtered


def make_trainer(SFTTrainer, *, model, args, train_dataset, eval_dataset, tokenizer):
    """Construct an SFTTrainer, handling the processing_class/tokenizer API rename."""
    supported = supported_params(SFTTrainer.__init__)
    kwargs = {"model": model, "args": args,
              "train_dataset": train_dataset, "eval_dataset": eval_dataset}
    if "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "Installed SFTTrainer accepts neither processing_class nor tokenizer"
        )
    return SFTTrainer(**kwargs)


# ---------------------------------------------------------------------------
# Checkpoint / persistence helpers
# ---------------------------------------------------------------------------

def config_manifest(repo_id: str, micro_batch_size: int, grad_accum: int):
    """Return a fingerprinted config dict used to guard against resume mismatches."""
    payload = {
        "run_tag":                    RUN_TAG,
        "model_name":                 MODEL_NAME,
        "model_revision":             MODEL_REVISION,
        "repo_id":                    repo_id,
        "max_length":                 MAX_LENGTH,
        "seed":                       SEED,
        "lora_r":                     LORA_R,
        "lora_alpha":                 LORA_ALPHA,
        "lora_dropout":               LORA_DROPOUT,
        "learning_rate":              LEARNING_RATE,
        "full_epochs":                FULL_EPOCHS,
        "micro_batch_size":           micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size":       micro_batch_size * grad_accum,
        "gradient_checkpointing":     USE_GRADIENT_CHECKPOINTING,
        "checkpoint_every_steps":     CHECKPOINT_EVERY_STEPS,
        "train_rows":                 25_000,
        "validation_rows":            2_000,
    }
    canonical        = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return payload


def read_json_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, data):
    """Atomic JSON write: write to .tmp then os.replace into final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def ensure_run_manifest(repo_id: str, micro_batch_size: int, grad_accum: int):
    """Write or verify the run manifest; raise if the config fingerprint changed."""
    expected = config_manifest(repo_id, micro_batch_size, grad_accum)
    if RUN_MANIFEST_PATH.exists():
        existing = read_json_file(RUN_MANIFEST_PATH)
        if existing.get("fingerprint") != expected["fingerprint"]:
            raise RuntimeError(
                "Existing checkpoint run was created with a different training "
                "configuration. Refusing to resume it.\n"
                f"Existing fingerprint: {existing.get('fingerprint')}\n"
                f"Current fingerprint : {expected['fingerprint']}\n"
                "Use a new RUN_TAG for a new experiment."
            )
        return existing
    write_json_file(RUN_MANIFEST_PATH, expected)
    checkpoint_volume.commit()
    return expected


def checkpoint_step(path: Path):
    """Extract the step number from a 'checkpoint-N' directory name, or -1."""
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def checkpoint_validation_errors(path: Path):
    """Return a list of error strings for a checkpoint dir, or [] if valid."""
    if not path.is_dir():
        return ["not a directory"]
    if checkpoint_step(path) < 0:
        return ["invalid checkpoint directory name"]

    errors = []
    if (path / "checkpoint-is-incomplete.txt").exists():
        errors.append("contains checkpoint-is-incomplete.txt")
    for fname in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json"]:
        if not (path / fname).is_file():
            errors.append(f"missing {fname}")
    if not ((path / "adapter_model.safetensors").is_file()
            or (path / "adapter_model.bin").is_file()):
        errors.append("missing adapter_model.safetensors/adapter_model.bin")
    return errors


def find_latest_valid_checkpoint():
    """Walk RUN_DIR descending by step; return the first valid checkpoint path."""
    if not RUN_DIR.exists():
        return None
    candidates = sorted(
        [p for p in RUN_DIR.iterdir() if p.is_dir() and checkpoint_step(p) >= 0],
        key=checkpoint_step, reverse=True,
    )
    for candidate in candidates:
        errors = checkpoint_validation_errors(candidate)
        if not errors:
            return str(candidate)
        print(f"Ignoring invalid checkpoint {candidate.name}: " + "; ".join(errors))
    return None


def final_adapter_errors(path: Path):
    """Return a list of error strings for a final adapter dir, or [] if valid."""
    if not path.is_dir():
        return ["final adapter directory is missing"]
    errors = []
    if not (path / "adapter_config.json").is_file():
        errors.append("missing adapter_config.json")
    if not ((path / "adapter_model.safetensors").is_file()
            or (path / "adapter_model.bin").is_file()):
        errors.append("missing adapter model weights")
    if not (path / "tokenizer_config.json").is_file():
        errors.append("missing tokenizer_config.json")
    return errors


def persist_final_adapter(model, tokenizer):
    """Save LoRA adapter + tokenizer to FINAL_ADAPTER_DIR via atomic staging swap."""
    staging = RUN_DIR / "final_adapter_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)

    errors = final_adapter_errors(staging)
    if errors:
        raise RuntimeError(
            "Final adapter validation failed before commit: " + "; ".join(errors)
        )
    if FINAL_ADAPTER_DIR.exists():
        shutil.rmtree(FINAL_ADAPTER_DIR)
    os.replace(staging, FINAL_ADAPTER_DIR)
    checkpoint_volume.commit()


def get_or_create_wandb_run_id():
    """Return a stable W&B run ID persisted across Modal replacement containers."""
    if WANDB_RUN_ID_PATH.exists():
        run_id = WANDB_RUN_ID_PATH.read_text(encoding="utf-8").strip()
        if run_id:
            return run_id
    run_id = uuid.uuid4().hex[:12]
    WANDB_RUN_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    WANDB_RUN_ID_PATH.write_text(run_id, encoding="utf-8")
    checkpoint_volume.commit()
    return run_id


def repair_final_adapter_metadata_for_hub():
    """
    PEFT may write MODEL_DIR (/models/qwen3-4b) as the base_model path in
    adapter_config.json; Hugging Face Hub rejects local filesystem paths.
    Patch adapter_config.json and README.md in-place before upload.
    """
    errors = final_adapter_errors(FINAL_ADAPTER_DIR)
    if errors:
        raise RuntimeError("Cannot repair final adapter metadata: " + "; ".join(errors))

    adapter_config_path = FINAL_ADAPTER_DIR / "adapter_config.json"
    adapter_config      = json.loads(adapter_config_path.read_text(encoding="utf-8"))

    # Replace any local path with the public HF model ID.
    adapter_config["base_model_name_or_path"] = MODEL_NAME
    adapter_config_path.write_text(
        json.dumps(adapter_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Replace any PEFT-generated model card that embeds the local MODEL_DIR.
    readme_path = FINAL_ADAPTER_DIR / "README.md"
    readme_path.write_text(
        "# PrivacyGuard Qwen3-4B QLoRA Adapter\n\n"
        f"Base model: `{MODEL_NAME}`\n\n"
        "QLoRA adapter fine-tuned for PII extraction and privacy-preserving redaction.\n",
        encoding="utf-8",
    )

    # Commit so retries/preemptions see the repaired files.
    checkpoint_volume.commit()


def upload_final_adapter(repo_id: str):
    """Repair metadata, push the adapter folder to the Hub, and verify the upload."""
    from huggingface_hub import HfApi

    errors = final_adapter_errors(FINAL_ADAPTER_DIR)
    if errors:
        raise RuntimeError("Cannot upload final adapter: " + "; ".join(errors))

    # Repair in-place first — a failed upload never requires retraining.
    repair_final_adapter_metadata_for_hub()

    token = os.environ["HF_TOKEN"]
    api   = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id, repo_type="model",
        folder_path=str(FINAL_ADAPTER_DIR), token=token,
        commit_message="Upload PrivacyGuard QLoRA adapter",
    )

    repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", token=token))
    required   = {"adapter_config.json", "tokenizer_config.json"}
    missing    = required - repo_files
    if not ("adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files):
        missing.add("adapter_model.safetensors/adapter_model.bin")
    if missing:
        raise RuntimeError(
            "Hub upload completed but required files are missing: "
            + ", ".join(sorted(missing))
        )
    return sorted(repo_files)


# ---------------------------------------------------------------------------
# CPU preflight  (validates data, deps, and SFTConfig before spending GPU $)
# ---------------------------------------------------------------------------

@app.function(
    image=image, cpu=4, memory=16384, timeout=3600,
    secrets=[secrets], volumes={CHECKPOINT_MOUNT: checkpoint_volume},
)
def preflight(smoke_test: bool = True, repo_id: str = "", micro_batch_size: int = 1):
    import accelerate, bitsandbytes, datasets, peft, torch, transformers, trl, wandb
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    grad_accum = resolve_grad_accum(micro_batch_size)

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is missing from .env")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is missing from .env")

    print("=" * 68)
    print("PrivacyGuard CPU preflight v6")
    print("=" * 68)
    print(f"torch        : {torch.__version__}")
    print(f"transformers : {transformers.__version__}")
    print(f"trl          : {trl.__version__}")
    print(f"peft         : {peft.__version__}")
    print(f"bitsandbytes : {bitsandbytes.__version__}")
    print(f"accelerate   : {accelerate.__version__}")
    print(f"datasets     : {datasets.__version__}")
    print(f"wandb        : {wandb.__version__}")
    print("=" * 68)

    hf_api   = HfApi(token=os.environ["HF_TOKEN"])
    identity = hf_api.whoami()
    print(f"Hugging Face identity: {identity.get('name', '<unknown>')}")

    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True, relogin=False)

    train_rows = read_jsonl(REMOTE_TRAIN)
    val_rows   = read_jsonl(REMOTE_VAL)

    if len(train_rows) != 25_000:
        raise RuntimeError(f"Expected 25,000 train rows; got {len(train_rows):,}")
    if len(val_rows) != 2_000:
        raise RuntimeError(f"Expected 2,000 validation rows; got {len(val_rows):,}")

    validate_rows(train_rows, "train")
    validate_rows(val_rows, "validation")

    if not smoke_test:
        if not repo_id:
            raise ValueError("Full training requires --repo-id USER/privacyguard-qwen3-4b-qlora")

        # Validate write permission before GPU allocation.
        hf_api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

        # Check whether this exact v6 run already completed.
        checkpoint_volume.reload()
        if COMPLETED_PATH.exists():
            completed = read_json_file(COMPLETED_PATH)
            if completed.get("repo_id") == repo_id:
                print("\nThis v6 training run is already complete.")
                return {"status": "already_complete", "completed": completed}

    check_train = train_rows[:256] if smoke_test else train_rows
    check_val   = val_rows[:64]    if smoke_test else val_rows

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_stats = length_stats(check_train, tokenizer, "Train preflight")
    val_stats   = length_stats(check_val,   tokenizer, "Validation preflight")

    # Refuse to proceed if more than 1 % of examples would be silently truncated.
    for name, stats in (("train", train_stats), ("validation", val_stats)):
        fraction = stats["over_limit"] / stats["count"] if stats["count"] else 0.0
        if fraction > 0.01:
            raise RuntimeError(
                f"{fraction:.2%} of {name} examples exceed MAX_LENGTH={MAX_LENGTH}. "
                "Refusing to silently truncate a meaningful fraction of the dataset."
            )

    sft_kwargs = make_sft_kwargs(
        SFTConfig, smoke_test=smoke_test, train_count=len(check_train),
        micro_batch_size=micro_batch_size, grad_accum=grad_accum, cpu_validation=True,
    )

    trainer_supported = supported_params(SFTTrainer.__init__)
    if "processing_class" not in trainer_supported and "tokenizer" not in trainer_supported:
        raise RuntimeError(
            "Unsupported SFTTrainer API: neither processing_class nor tokenizer"
        )

    result = {
        "status":                   "ok",
        "smoke_test":               smoke_test,
        "train_rows":               len(train_rows),
        "validation_rows":          len(val_rows),
        "micro_batch_size":         micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size":     micro_batch_size * grad_accum,
        "gradient_checkpointing":   USE_GRADIENT_CHECKPOINTING,
        "warmup_steps":             sft_kwargs["warmup_steps"],
        "train_length_stats":       train_stats,
        "validation_length_stats":  val_stats,
        "run_tag":                  RUN_TAG,
    }
    print("\nCPU PREFLIGHT PASSED")
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# GPU training
#
# Follows Modal's long-training pattern:
#   - persistent Volume for durable checkpoints
#   - retries=10 with single_use_containers so each retry is a clean container
#   - DurableCheckpointCallback commits every Trainer save to the Volume
# ---------------------------------------------------------------------------

@app.function(
    image=image, gpu="L4", cpu=4, memory=32768,
    timeout=12 * 60 * 60,
    secrets=[secrets], volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    single_use_containers=True,
)
def train(smoke_test: bool = True, repo_id: str = "", micro_batch_size: int = 1):
    import torch, wandb
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Modal GPU container")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The allocated GPU does not support BF16")

    grad_accum = resolve_grad_accum(micro_batch_size)

    train_rows = read_jsonl(REMOTE_TRAIN)
    val_rows   = read_jsonl(REMOTE_VAL)

    if smoke_test:
        train_rows = train_rows[:256]
        val_rows   = val_rows[:64]
    else:
        if len(train_rows) != 25_000 or len(val_rows) != 2_000:
            raise RuntimeError("Processed train/validation dataset sizes changed")
        if not repo_id:
            raise ValueError("Full training requires --repo-id")

        # Make latest committed state visible in a replacement container.
        checkpoint_volume.reload()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        ensure_run_manifest(repo_id, micro_batch_size, grad_accum)

        # If a valid final adapter already exists, skip training and retry upload only.
        if not final_adapter_errors(FINAL_ADAPTER_DIR):
            print(
                "\nValid final adapter already exists on the durable Volume. "
                "Skipping training and retrying Hub upload only."
            )
            repo_files     = upload_final_adapter(repo_id)
            stored_summary = read_json_file(TRAINING_SUMMARY_PATH) if TRAINING_SUMMARY_PATH.exists() else {}
            completed = {
                **stored_summary,
                "status": "complete", "repo_id": repo_id,
                "hub_files": repo_files, "run_tag": RUN_TAG,
            }
            write_json_file(COMPLETED_PATH, completed)
            checkpoint_volume.commit()
            return completed

    print("=" * 68)
    print("PrivacyGuard QLoRA training v6.1 (upload metadata fix)")
    print("=" * 68)
    print(f"GPU                    : {torch.cuda.get_device_name(0)}")
    print(f"Smoke test             : {smoke_test}")
    print(f"Train samples          : {len(train_rows):,}")
    print(f"Validation samples     : {len(val_rows):,}")
    print(f"Micro batch            : {micro_batch_size}")
    print(f"Gradient accumulation  : {grad_accum}")
    print(f"Effective batch        : {micro_batch_size * grad_accum}")
    print(f"Gradient checkpointing : {USE_GRADIENT_CHECKPOINTING}")
    print(f"Run tag                : {RUN_TAG}")
    print("=" * 68)

    # Validate exact real GPU config before loading the 4B model.
    sft_kwargs = make_sft_kwargs(
        SFTConfig, smoke_test=smoke_test, train_count=len(train_rows),
        micro_batch_size=micro_batch_size, grad_accum=grad_accum, cpu_validation=False,
    )
    args = SFTConfig(**sft_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = build_dataset(train_rows, tokenizer)
    val_dataset   = build_dataset(val_rows,   tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading Qwen3-4B in 4-bit NF4...")
    load_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=bnb_config,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
        ),
    )
    model_load_seconds = time.perf_counter() - load_start
    model.print_trainable_parameters()

    last_checkpoint = None
    wandb_run_id    = None

    if not smoke_test:
        last_checkpoint = find_latest_valid_checkpoint()
        if last_checkpoint:
            print(f"\nRESUMING from durable checkpoint: {last_checkpoint}")
        else:
            print("\nNo valid durable checkpoint found; starting from step 0.")

        wandb_run_id = get_or_create_wandb_run_id()
        # Keep one W&B run across Modal replacement containers.
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_NAME"]   = RUN_TAG

    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True, relogin=False)

    trainer = make_trainer(
        SFTTrainer, model=model, args=args,
        train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=tokenizer,
    )

    if not smoke_test:
        class DurableCheckpointCallback(TrainerCallback):
            """Commit each Trainer checkpoint to the Modal Volume for durability."""
            def on_save(self, args, state, control, **kwargs):
                candidate = RUN_DIR / f"checkpoint-{state.global_step}"
                errors    = checkpoint_validation_errors(candidate)
                if errors:
                    print(
                        f"WARNING: checkpoint-{state.global_step} was not committed "
                        "because validation failed: " + "; ".join(errors)
                    )
                    return control
                checkpoint_volume.commit()
                print(f"Durable checkpoint committed at global_step={state.global_step}")
                return control

        trainer.add_callback(DurableCheckpointCallback())

    torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()

    try:
        train_result = trainer.train(
            resume_from_checkpoint=(last_checkpoint if not smoke_test and last_checkpoint else None)
        )
        training_seconds = time.perf_counter() - training_start
        eval_metrics     = trainer.evaluate()

        peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb  = torch.cuda.max_memory_reserved()  / (1024 ** 3)

        summary = {
            "status":                    "trained",
            "smoke_test":                smoke_test,
            "run_tag":                   RUN_TAG,
            "train_samples":             len(train_dataset),
            "validation_samples":        len(val_dataset),
            "micro_batch_size":          micro_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size":      micro_batch_size * grad_accum,
            "gradient_checkpointing":    USE_GRADIENT_CHECKPOINTING,
            "model_load_seconds":        model_load_seconds,
            "training_seconds":          training_seconds,
            "train_samples_per_second":  train_result.metrics.get("train_samples_per_second"),
            "train_steps_per_second":    train_result.metrics.get("train_steps_per_second"),
            "train_loss":                train_result.metrics.get("train_loss"),
            "eval_loss":                 eval_metrics.get("eval_loss"),
            "peak_cuda_allocated_gb":    peak_allocated_gb,
            "peak_cuda_reserved_gb":     peak_reserved_gb,
            "repo_id":                   None if smoke_test else repo_id,
            "resumed_from_checkpoint":   last_checkpoint,
            "wandb_run_id":              wandb_run_id,
        }

        if not smoke_test:
            # Persist the final adapter BEFORE network upload.  If Hub upload is
            # interrupted, a retry sees this directory and only retries upload.
            persist_final_adapter(trainer.model, tokenizer)
            write_json_file(TRAINING_SUMMARY_PATH, summary)
            checkpoint_volume.commit()

            print(f"\nUploading final adapter to {repo_id}...")
            repo_files = upload_final_adapter(repo_id)

            completed = {**summary, "status": "complete", "hub_files": repo_files}
            write_json_file(COMPLETED_PATH, completed)
            checkpoint_volume.commit()
            summary = completed

        print("\n" + "=" * 68)
        print("TRAINING SUMMARY")
        print("=" * 68)
        print(json.dumps(summary, indent=2))
        print("=" * 68)
        return summary

    finally:
        wandb.finish()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(smoke_test: bool = True, repo_id: str = "", micro_batch_size: int = 1):
    print("\nSTEP 1/2: CPU preflight")

    preflight_result = preflight.remote(
        smoke_test=smoke_test, repo_id=repo_id, micro_batch_size=micro_batch_size,
    )

    if preflight_result.get("status") == "already_complete":
        print("\nTraining was already completed successfully.")
        print(json.dumps(preflight_result["completed"], indent=2))
        return

    if preflight_result.get("status") != "ok":
        raise RuntimeError("CPU preflight failed")

    print("\nSTEP 2/2: Starting resumable L4 job")

    # Modal's long-training guide recommends spawn(...).get() for durable,
    # detached long-running training calls.
    function_call = train.spawn(
        smoke_test=smoke_test, repo_id=repo_id, micro_batch_size=micro_batch_size,
    )
    print(f"Training FunctionCall ID: {function_call.object_id}")

    summary = function_call.get()
    print("\nReturned summary:")
    print(json.dumps(summary, indent=2))
