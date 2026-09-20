import json
import os
import sys
from pathlib import Path

from datasets import load_dataset


# =========================================================
# Windows UTF-8
# =========================================================

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load .env
# =========================================================

ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    with ENV_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# =========================================================
# Configuration
# =========================================================

DATASET_NAME = "ai4privacy/pii-masking-openpii-1m"

SEED = 42
TRAIN_SIZE = 25_000
VALIDATION_SIZE = 2_000
TEST_SIZE = 3_000
SHUFFLE_BUFFER = 20_000


# =========================================================
# Validation / conversion
# =========================================================

def convert_example(example):
    """
    Convert an OpenPII record into the model-independent
    PrivacyGuard format.

    Output:

    {
        "uid": "...",
        "text": "...",
        "entities": [
            {
                "text": "...",
                "type": "...",
                "start": 10,
                "end": 20
            }
        ]
    }
    """

    text = example.get("source_text")

    if not isinstance(text, str):
        return None

    entities = []

    for item in example.get("privacy_mask", []):
        value = item.get("value")
        label = item.get("label")
        start = item.get("start")
        end = item.get("end")

        # Basic structure validation
        if not isinstance(value, str):
            continue
        if not isinstance(label, str):
            continue
        if not isinstance(start, int):
            continue
        if not isinstance(end, int):
            continue
        if not (0 <= start < end <= len(text)):
            continue

        # Critical span validation
        if text[start:end] != value:
            raise ValueError(
                "\nInvalid PII span detected\n"
                f"UID: {example.get('uid')}\n"
                f"Expected: {value!r}\n"
                f"Actual: {text[start:end]!r}\n"
                f"Start: {start}\n"
                f"End: {end}"
            )

        entities.append({"text": value, "type": label, "start": start, "end": end})

    return {
        "uid": str(example.get("uid", "")),
        "text": text,
        "entities": entities,
    }


# =========================================================
# Save JSONL
# =========================================================

def save_jsonl(records, output_path):
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records):,} records -> {output_path}")


# =========================================================
# Collect English records
# =========================================================

def collect_english(dataset, target_count, name):
    """
    Collect exactly target_count English records.
    """

    records = []

    for example in dataset:
        # OpenPII language field
        language = str(example.get("language", "")).lower()

        if language not in {"en", "english"}:
            continue

        converted = convert_example(example)
        if converted is None:
            continue

        records.append(converted)
        count = len(records)

        if count % 1000 == 0 or count == target_count:
            print(f"{name}: {count:,}/{target_count:,}")

        if count >= target_count:
            break

    if len(records) < target_count:
        raise RuntimeError(
            f"Only found {len(records):,} English records for {name}. "
            f"Expected {target_count:,}."
        )

    return records


# =========================================================
# Main
# =========================================================

def main():
    print("\nPrivacyGuard dataset preparation")
    print("=" * 50)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Seed: {SEED}")
    print(f"Train: {TRAIN_SIZE:,}")
    print(f"Validation: {VALIDATION_SIZE:,}")
    print(f"Test: {TEST_SIZE:,}")
    print("=" * 50)

    # -----------------------------------------------------
    # TRAIN + VALIDATION
    #
    # Both come from official OpenPII train split.
    # We iterate ONCE so the two subsets cannot overlap.
    # -----------------------------------------------------

    print("\nLoading official OpenPII train split...")

    train_stream = load_dataset(DATASET_NAME, split="train", streaming=True)
    train_stream = train_stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)

    combined_needed = TRAIN_SIZE + VALIDATION_SIZE
    combined = collect_english(train_stream, combined_needed, "Train + Validation")

    train_records = combined[:TRAIN_SIZE]
    validation_records = combined[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]

    # -----------------------------------------------------
    # TEST
    #
    # Comes ONLY from official OpenPII validation split.
    # -----------------------------------------------------

    print("\nLoading official OpenPII validation split...")

    test_stream = load_dataset(DATASET_NAME, split="validation", streaming=True)
    test_stream = test_stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)

    test_records = collect_english(test_stream, TEST_SIZE, "Test")

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    assert len(train_records) == TRAIN_SIZE
    assert len(validation_records) == VALIDATION_SIZE
    assert len(test_records) == TEST_SIZE

    train_ids = {record["uid"] for record in train_records}
    validation_ids = {record["uid"] for record in validation_records}
    test_ids = {record["uid"] for record in test_records}

    if train_ids & validation_ids:
        raise RuntimeError("Train/validation UID overlap detected.")

    # Official validation should already be separate,
    # but verify anyway.
    if train_ids & test_ids:
        raise RuntimeError("Train/test UID overlap detected.")

    if validation_ids & test_ids:
        raise RuntimeError("Validation/test UID overlap detected.")

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print("\nSaving processed datasets...")

    save_jsonl(train_records, PROCESSED_DIR / "train.jsonl")
    save_jsonl(validation_records, PROCESSED_DIR / "validation.jsonl")
    save_jsonl(test_records, PROCESSED_DIR / "test.jsonl")

    print("\n" + "=" * 50)
    print("Dataset preparation complete")
    print("=" * 50)
    print(f"Train       : {len(train_records):,}")
    print(f"Validation  : {len(validation_records):,}")
    print(f"Test        : {len(test_records):,}")
    print("\nThese files should now be frozen for training/evaluation.")


if __name__ == "__main__":
    main()