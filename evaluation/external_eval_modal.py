import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import modal


# =========================================================
# Paths / configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_GRETEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "gretel_test_1000.jsonl"
)

LOCAL_NEMOTRON_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "nemotron_test_1000.jsonl"
)

REMOTE_GRETEL_FILE = "/data/gretel_test_1000.jsonl"
REMOTE_NEMOTRON_FILE = "/data/nemotron_test_1000.jsonl"

MODEL_NAME = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7"
MODEL_DIR = "/models/qwen3-4b"

ADAPTER_REPO = "rohitkmr8527/privacyguard-qwen3-4b-qlora"
ADAPTER_DIR = "/models/privacyguard-qwen3-4b-qlora"
LORA_RANK = 16

MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 384
GPU_MEMORY_UTILIZATION = 0.88


# =========================================================
# PrivacyGuard taxonomy
# =========================================================

ALLOWED_TYPES = frozenset(
    {
        "DATE",
        "GIVENNAME",
        "SURNAME",
        "EMAIL",
        "CITY",
        "TITLE",
        "TELEPHONENUM",
        "AGE",
        "STREET",
        "BUILDINGNUM",
        "ZIPCODE",
        "IDCARDNUM",
        "CREDITCARDNUMBER",
        "DRIVERLICENSENUM",
        "GENDER",
        "TAXNUM",
        "SEX",
        "SOCIALNUM",
        "PASSPORTNUM",
    }
)


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

app = modal.App("privacyguard-external-eval")

hf_secret = modal.Secret.from_dotenv(
    PROJECT_ROOT,
    filename=".env",
)


def download_assets(
    model_name: str,
    model_revision: str,
    model_dir: str,
    adapter_repo: str,
    adapter_dir: str,
):
    import json
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError("HF_TOKEN is missing.")

    api = HfApi(token=token)

    adapter_info = api.model_info(
        repo_id=adapter_repo,
        token=token,
    )
    adapter_revision = adapter_info.sha

    if not adapter_revision:
        raise RuntimeError(
            f"Could not resolve adapter revision for {adapter_repo}."
        )

    print(
        f"Resolved adapter {adapter_repo} "
        f"revision={adapter_revision}"
    )

    snapshot_download(
        repo_id=model_name,
        revision=model_revision,
        local_dir=model_dir,
        token=token,
    )

    snapshot_download(
        repo_id=adapter_repo,
        revision=adapter_revision,
        local_dir=adapter_dir,
        token=token,
    )

    config_path = (
        Path(adapter_dir)
        / "adapter_config.json"
    )

    if not config_path.is_file():
        raise RuntimeError(
            "adapter_config.json is missing."
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        adapter_config = json.load(file)

    rank = adapter_config.get("r")

    if rank != LORA_RANK:
        raise RuntimeError(
            f"Unexpected LoRA rank: {rank!r}"
        )

    # Normalize only the image-local adapter metadata when needed.
    accepted_base_names = {
        model_name,
        model_dir,
        "/models/qwen3-4b",
    }

    adapter_base = adapter_config.get(
        "base_model_name_or_path"
    )

    if adapter_base not in accepted_base_names:
        raise RuntimeError(
            "Adapter/base-model mismatch: "
            f"{adapter_base!r}"
        )

    if adapter_base != model_name:
        adapter_config[
            "base_model_name_or_path"
        ] = model_name

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                adapter_config,
                file,
                indent=2,
                ensure_ascii=False,
            )

    (
        Path(adapter_dir)
        / "_resolved_revision.txt"
    ).write_text(
        adapter_revision,
        encoding="utf-8",
    )


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .run_function(
        download_assets,
        args=(
            MODEL_NAME,
            MODEL_REVISION,
            MODEL_DIR,
            ADAPTER_REPO,
            ADAPTER_DIR,
        ),
        secrets=[hf_secret],
        cpu=4,
        memory=16384,
        timeout=60 * 60,
    )
    .add_local_file(
        str(LOCAL_GRETEL_FILE),
        REMOTE_GRETEL_FILE,
    )
    .add_local_file(
        str(LOCAL_NEMOTRON_FILE),
        REMOTE_NEMOTRON_FILE,
    )
)


# =========================================================
# Helpers
# =========================================================

def load_jsonl(path):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"line {line_number}"
                ) from exc

            missing = {
                "uid",
                "text",
                "entities",
                "ignored_entities",
            } - set(row)

            if missing:
                raise ValueError(
                    f"{path} line {line_number} "
                    f"missing {sorted(missing)}"
                )

            rows.append(row)

    if not rows:
        raise ValueError(
            f"No rows loaded from {path}"
        )

    return rows


def parse_response(raw_text):
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)

        if (
            isinstance(parsed, dict)
            and isinstance(
                parsed.get("entities"),
                list,
            )
        ):
            return (
                parsed["entities"],
                True,
            )
    except (
        json.JSONDecodeError,
        TypeError,
    ):
        pass

    # External evaluation is intentionally strict.
    # Invalid raw JSON means an empty prediction.
    return [], False


def align_entities_to_text(
    source_text,
    predicted_entities,
):
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

        if label not in ALLOWED_TYPES:
            continue

        position = source_text.find(value)

        while position != -1:
            start = position
            end = position + len(value)
            key = (
                start,
                end,
                label,
            )

            if key not in used_spans:
                used_spans.add(key)

                aligned.append(
                    {
                        "text": value,
                        "type": label,
                        "start": start,
                        "end": end,
                    }
                )
                break

            position = source_text.find(
                value,
                position + 1,
            )

    return aligned


def overlaps_ignored(
    predicted_entity,
    ignored_entities,
):
    """
    Shared-taxonomy scoring:
    predictions on spans annotated only with an unsupported external
    label are excluded instead of counted as false positives.
    """
    pred_start = int(
        predicted_entity["start"]
    )
    pred_end = int(
        predicted_entity["end"]
    )

    for ignored in ignored_entities:
        ignored_start = int(
            ignored["start"]
        )
        ignored_end = int(
            ignored["end"]
        )

        if (
            pred_start < ignored_end
            and pred_end > ignored_start
        ):
            return True

    return False


def calculate_metrics(results):
    tp = fp = fn = 0
    valid_json_count = 0
    ignored_prediction_count = 0

    by_type = {
        label: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for label in sorted(
            ALLOWED_TYPES
        )
    }

    for row in results:
        gold = {
            (
                int(entity["start"]),
                int(entity["end"]),
                str(entity["type"]).upper(),
            )
            for entity in row[
                "gold_entities"
            ]
        }

        if row.get(
            "valid_json",
            False,
        ):
            valid_json_count += 1

            scored_predictions = []

            for entity in row[
                "predicted_entities"
            ]:
                if overlaps_ignored(
                    entity,
                    row["ignored_entities"],
                ):
                    ignored_prediction_count += 1
                    continue

                scored_predictions.append(
                    entity
                )

            predicted = {
                (
                    int(entity["start"]),
                    int(entity["end"]),
                    str(entity["type"]).upper(),
                )
                for entity
                in scored_predictions
            }
        else:
            predicted = set()

        tp_set = gold & predicted
        fp_set = predicted - gold
        fn_set = gold - predicted

        tp += len(tp_set)
        fp += len(fp_set)
        fn += len(fn_set)

        for _, _, label in tp_set:
            by_type[label]["tp"] += 1

        for _, _, label in fp_set:
            by_type[label]["fp"] += 1

        for _, _, label in fn_set:
            by_type[label]["fn"] += 1

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )
    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )
    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    beta = 2.0
    f2 = (
        (1 + beta**2)
        * precision
        * recall
        / (
            beta**2
            * precision
            + recall
        )
        if (
            beta**2
            * precision
            + recall
        )
        else 0.0
    )

    per_entity = {}

    for label, counts in by_type.items():
        label_tp = counts["tp"]
        label_fp = counts["fp"]
        label_fn = counts["fn"]

        support = label_tp + label_fn

        if (
            support == 0
            and label_fp == 0
        ):
            continue

        p = (
            label_tp
            / (label_tp + label_fp)
            if label_tp + label_fp
            else 0.0
        )
        r = (
            label_tp
            / (label_tp + label_fn)
            if label_tp + label_fn
            else 0.0
        )
        label_f1 = (
            2 * p * r
            / (p + r)
            if p + r
            else 0.0
        )

        per_entity[label] = {
            "precision": p,
            "recall": r,
            "f1": label_f1,
            "support": support,
            "true_positives": label_tp,
            "false_positives": label_fp,
            "false_negatives": label_fn,
        }

    macro_f1 = (
        sum(
            item["f1"]
            for item
            in per_entity.values()
        )
        / len(per_entity)
        if per_entity
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "pii_leakage_rate": (
            fn / (tp + fn)
            if tp + fn
            else 0.0
        ),
        "over_redaction_rate": (
            fp / (tp + fp)
            if tp + fp
            else 0.0
        ),
        "json_validity": (
            valid_json_count
            / len(results)
            if results
            else 0.0
        ),
        "macro_f1": macro_f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "ignored_predictions": (
            ignored_prediction_count
        ),
        "by_entity": per_entity,
    }


def build_conversations(samples):
    return [
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Extract all PII "
                    "from this text:\n\n"
                    + sample["text"]
                ),
            },
        ]
        for sample in samples
    ]


def run_batch(
    llm,
    sampling_params,
    samples,
    lora_request,
    label,
):
    print(
        f"\nStarting {label} "
        f"({len(samples)} samples)..."
    )

    conversations = (
        build_conversations(samples)
    )

    start_time = time.perf_counter()

    outputs = llm.chat(
        conversations,
        sampling_params=(
            sampling_params
        ),
        lora_request=(
            lora_request
        ),
        use_tqdm=True,
        chat_template_kwargs={
            "enable_thinking": False,
        },
    )

    generation_seconds = (
        time.perf_counter()
        - start_time
    )

    results = []
    finish_reasons = Counter()
    total_output_tokens = 0

    for sample, output in zip(
        samples,
        outputs,
    ):
        completion = output.outputs[0]

        raw_response = (
            completion.text.strip()
        )

        finish_reason = (
            completion.finish_reason
            or "unknown"
        )

        finish_reasons[
            finish_reason
        ] += 1

        total_output_tokens += len(
            completion.token_ids
        )

        raw_entities, valid_json = (
            parse_response(
                raw_response
            )
        )

        predicted_entities = (
            align_entities_to_text(
                sample["text"],
                raw_entities,
            )
        )

        results.append(
            {
                "uid": sample["uid"],
                "source_dataset":
                    sample[
                        "source_dataset"
                    ],
                "text": sample["text"],
                "gold_entities":
                    sample["entities"],
                "ignored_entities":
                    sample[
                        "ignored_entities"
                    ],
                "predicted_entities":
                    predicted_entities,
                "valid_json":
                    valid_json,
                "finish_reason":
                    finish_reason,
                "raw_response":
                    raw_response,
            }
        )

    return {
        "results": results,
        "timing": {
            "generation_seconds":
                generation_seconds,
            "samples_per_second":
                len(samples)
                / generation_seconds,
            "total_output_tokens":
                total_output_tokens,
            "output_tokens_per_second":
                (
                    total_output_tokens
                    / generation_seconds
                    if generation_seconds
                    else 0.0
                ),
            "finish_reasons":
                dict(
                    finish_reasons
                ),
        },
    }


# =========================================================
# Remote evaluation
# =========================================================

@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24576,
    timeout=60 * 60,
)
def run_external_evaluation():
    from vllm import (
        LLM,
        SamplingParams,
    )
    from vllm.lora.request import (
        LoRARequest,
    )

    gretel = load_jsonl(
        REMOTE_GRETEL_FILE
    )
    nemotron = load_jsonl(
        REMOTE_NEMOTRON_FILE
    )

    if len(gretel) != 1000:
        raise RuntimeError(
            "Expected 1000 Gretel samples."
        )

    if len(nemotron) != 1000:
        raise RuntimeError(
            "Expected 1000 Nemotron samples."
        )

    print(
        "\nLoading Qwen3-4B once "
        "for base + LoRA evaluation..."
    )

    load_start = time.perf_counter()

    llm = LLM(
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=(
            GPU_MEMORY_UTILIZATION
        ),
        enable_prefix_caching=True,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_loras=1,
    )

    model_load_seconds = (
        time.perf_counter()
        - load_start
    )

    adapter_revision = (
        (
            Path(ADAPTER_DIR)
            / "_resolved_revision.txt"
        )
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    lora_request = LoRARequest(
        "privacyguard-pii",
        1,
        ADAPTER_DIR,
    )

    sampling_params = (
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS,
            seed=42,
        )
    )

    evaluations = {}

    for dataset_name, samples in [
        ("gretel", gretel),
        ("nemotron", nemotron),
    ]:
        base_payload = run_batch(
            llm,
            sampling_params,
            samples,
            None,
            f"{dataset_name} / base",
        )

        finetuned_payload = run_batch(
            llm,
            sampling_params,
            samples,
            lora_request,
            f"{dataset_name} / fine-tuned",
        )

        base_payload["metrics"] = (
            calculate_metrics(
                base_payload["results"]
            )
        )

        finetuned_payload[
            "metrics"
        ] = calculate_metrics(
            finetuned_payload[
                "results"
            ]
        )

        evaluations[
            dataset_name
        ] = {
            "base":
                base_payload,
            "finetuned":
                finetuned_payload,
        }

    return {
        "model": MODEL_NAME,
        "model_revision":
            MODEL_REVISION,
        "adapter_repo":
            ADAPTER_REPO,
        "adapter_revision":
            adapter_revision,
        "model_load_seconds":
            model_load_seconds,
        "evaluations":
            evaluations,
    }


# =========================================================
# Local entrypoint
# =========================================================

def print_summary(
    dataset_name,
    model_name,
    metrics,
):
    print(
        f"{dataset_name:10s} "
        f"{model_name:10s} "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"F2={metrics['f2']:.4f}  "
        f"Leak={metrics['pii_leakage_rate']:.4f}  "
        f"JSON={metrics['json_validity']:.4f}"
    )


@app.local_entrypoint()
def main():
    total_start = time.perf_counter()

    payload = (
        run_external_evaluation.remote()
    )

    round_trip_seconds = (
        time.perf_counter()
        - total_start
    )

    results_dir = (
        PROJECT_ROOT
        / "results"
        / "external"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = {
        "configuration": {
            "model":
                payload["model"],
            "model_revision":
                payload[
                    "model_revision"
                ],
            "adapter_repo":
                payload[
                    "adapter_repo"
                ],
            "adapter_revision":
                payload[
                    "adapter_revision"
                ],
            "scoring": (
                "Strict exact span + type "
                "on shared taxonomy. "
                "Predictions overlapping "
                "unsupported external gold "
                "spans are excluded."
            ),
        },
        "model_load_seconds":
            payload[
                "model_load_seconds"
            ],
        "round_trip_seconds":
            round_trip_seconds,
        "datasets": {},
    }

    for dataset_name, evaluation in (
        payload[
            "evaluations"
        ].items()
    ):
        comparison["datasets"][
            dataset_name
        ] = {}

        for model_name in [
            "base",
            "finetuned",
        ]:
            current = (
                evaluation[
                    model_name
                ]
            )

            predictions_file = (
                results_dir
                / (
                    f"{dataset_name}_"
                    f"{model_name}_"
                    f"predictions.jsonl"
                )
            )

            with predictions_file.open(
                "w",
                encoding="utf-8",
            ) as file:
                for row in current[
                    "results"
                ]:
                    file.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            comparison["datasets"][
                dataset_name
            ][model_name] = {
                "metrics":
                    current["metrics"],
                "timing":
                    current["timing"],
                "predictions_file":
                    str(
                        predictions_file
                        .relative_to(
                            PROJECT_ROOT
                        )
                    ),
            }

    output_file = (
        results_dir
        / "external_benchmark_metrics.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "EXTERNAL BENCHMARK SUMMARY"
    )
    print(
        "=" * 100
    )

    for dataset_name in [
        "gretel",
        "nemotron",
    ]:
        for model_name in [
            "base",
            "finetuned",
        ]:
            metrics = (
                comparison[
                    "datasets"
                ][dataset_name][
                    model_name
                ]["metrics"]
            )

            print_summary(
                dataset_name,
                model_name,
                metrics,
            )

    print(
        f"\nMetrics -> "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
