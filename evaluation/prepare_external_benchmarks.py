import ast
import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "external"

SEED = 42
SAMPLE_SIZE = 1000

GRETEL_REPO = "gretelai/gretel-pii-masking-en-v1"
NEMOTRON_REPO = "nvidia/Nemotron-PII"

# Map only labels with a clear equivalent in PrivacyGuard's
# 19-label OpenPII taxonomy. Everything else stays ignored.
LABEL_MAP = {
    "first_name": "GIVENNAME",
    "given_name": "GIVENNAME",
    "last_name": "SURNAME",
    "surname": "SURNAME",

    "email": "EMAIL",
    "email_address": "EMAIL",

    "phone_number": "TELEPHONENUM",
    "telephone_number": "TELEPHONENUM",

    "date": "DATE",
    "date_of_birth": "DATE",
    "birth_date": "DATE",
    "date_time": "DATE",

    "age": "AGE",

    "city": "CITY",

    "postcode": "ZIPCODE",
    "postal_code": "ZIPCODE",
    "zip_code": "ZIPCODE",

    "credit_card_number": "CREDITCARDNUMBER",

    "ssn": "SOCIALNUM",
    "social_security_number": "SOCIALNUM",

    "tax_id": "TAXNUM",
    "tax_number": "TAXNUM",

    "national_id": "IDCARDNUM",
    "id_card_number": "IDCARDNUM",

    "passport_number": "PASSPORTNUM",

    "driver_license_number": "DRIVERLICENSENUM",
    "drivers_license_number": "DRIVERLICENSENUM",
    "driving_license_number": "DRIVERLICENSENUM",

    "gender": "GENDER",
    "sex": "SEX",

    "title": "TITLE",
}

ALLOWED_TYPES = set(LABEL_MAP.values())


def load_local_hf_token():
    """
    Read HF_TOKEN from environment or project .env without adding
    python-dotenv as another dependency.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    env_path = PROJECT_ROOT / ".env"

    if not env_path.is_file():
        return None

    with env_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or "=" not in line
            ):
                continue

            key, value = line.split("=", 1)

            if key.strip() == "HF_TOKEN":
                token = value.strip().strip('"').strip("'")

                if token:
                    os.environ["HF_TOKEN"] = token
                    return token

    return None


HF_TOKEN = load_local_hf_token()


def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def parse_serialized(value):
    """
    Hugging Face may return these annotation columns either as
    real Python lists or as strings.

    Gretel stores entities like:
        "[{'entity': 'x@y.com', 'types': ['email']}]"

    That is Python-literal syntax, not valid JSON. Try JSON first,
    then safely parse Python literals.
    """
    if not isinstance(value, str):
        return value

    value = value.strip()

    if not value:
        return []

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            "Could not parse serialized annotation field: "
            f"{value[:160]!r}"
        ) from exc


def all_exact_occurrences(text, value):
    if not value:
        return []

    spans = []
    start = text.find(value)

    while start != -1:
        end = start + len(value)
        spans.append((start, end))
        start = text.find(value, start + 1)

    return spans


def dedupe_entities(entities):
    seen = set()
    output = []

    for entity in sorted(
        entities,
        key=lambda x: (
            int(x["start"]),
            int(x["end"]),
            str(
                x.get(
                    "type",
                    x.get("source_type", ""),
                )
            ),
        ),
    ):
        key = (
            int(entity["start"]),
            int(entity["end"]),
            str(entity.get("type", "")),
            str(entity.get("source_type", "")),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(entity)

    return output


def convert_gretel_row(row):
    text = str(row["text"])

    raw_entities = parse_serialized(
        row.get("entities", [])
    )

    if not isinstance(raw_entities, list):
        return None, "entities_not_list"

    mapped = []
    ignored = []

    for raw in raw_entities:
        if not isinstance(raw, dict):
            return None, "entity_not_object"

        value = raw.get("entity")
        types = parse_serialized(
            raw.get("types", [])
        )

        if isinstance(types, str):
            types = [types]

        if not isinstance(value, str) or not value:
            return None, "missing_entity_text"

        if not isinstance(types, (list, tuple)) or not types:
            return None, "missing_entity_type"

        occurrences = all_exact_occurrences(
            text,
            value,
        )

        if not occurrences:
            # Do not keep a document with incomplete alignment.
            return None, "unaligned_entity"

        normalized_types = [
            normalize_label(label)
            for label in types
        ]

        mapped_types = {
            LABEL_MAP[label]
            for label in normalized_types
            if label in LABEL_MAP
        }

        # Use as gold only when source labels map to exactly one
        # PrivacyGuard type.
        target_type = (
            next(iter(mapped_types))
            if len(mapped_types) == 1
            else None
        )

        source_type = "|".join(
            normalized_types
        )

        for start, end in occurrences:
            if target_type:
                mapped.append(
                    {
                        "text": text[start:end],
                        "type": target_type,
                        "start": start,
                        "end": end,
                        "source_type": source_type,
                    }
                )
            else:
                ignored.append(
                    {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "source_type": source_type,
                    }
                )

    mapped = dedupe_entities(mapped)
    ignored = dedupe_entities(ignored)

    if not mapped:
        return None, "no_mapped_entities"

    return {
        "uid": f"gretel:{row['uid']}",
        "source_dataset": GRETEL_REPO,
        "source_uid": str(row["uid"]),
        "domain": row.get("domain"),
        "document_type": row.get(
            "document_type"
        ),
        "text": text,
        "entities": mapped,
        "ignored_entities": ignored,
    }, None


def convert_nemotron_row(row):
    text = str(row["text"])

    raw_spans = parse_serialized(
        row.get("spans", [])
    )

    if not isinstance(raw_spans, list):
        return None, "spans_not_list"

    mapped = []
    ignored = []

    for span in raw_spans:
        if not isinstance(span, dict):
            return None, "span_not_object"

        if not {
            "start",
            "end",
            "label",
        } <= set(span):
            return None, "missing_span_fields"

        start = int(span["start"])
        end = int(span["end"])

        if not (
            0 <= start < end <= len(text)
        ):
            return None, "invalid_span"

        value = text[start:end]

        provided_text = span.get("text")

        if (
            isinstance(provided_text, str)
            and provided_text
            and provided_text != value
        ):
            return None, "span_text_mismatch"

        source_label = normalize_label(
            span["label"]
        )

        target_type = LABEL_MAP.get(
            source_label
        )

        if target_type:
            mapped.append(
                {
                    "text": value,
                    "type": target_type,
                    "start": start,
                    "end": end,
                    "source_type": source_label,
                }
            )
        else:
            ignored.append(
                {
                    "text": value,
                    "start": start,
                    "end": end,
                    "source_type": source_label,
                }
            )

    mapped = dedupe_entities(mapped)
    ignored = dedupe_entities(ignored)

    if not mapped:
        return None, "no_mapped_entities"

    return {
        "uid": f"nemotron:{row['uid']}",
        "source_dataset": NEMOTRON_REPO,
        "source_uid": str(row["uid"]),
        "domain": row.get("domain"),
        "document_type": row.get(
            "document_type"
        ),
        "document_format": row.get(
            "document_format"
        ),
        "locale": row.get("locale"),
        "text": text,
        "entities": mapped,
        "ignored_entities": ignored,
    }, None


def resolve_revision(repo_id):
    api = HfApi(token=HF_TOKEN)

    info = api.dataset_info(
        repo_id=repo_id,
    )

    if not info.sha:
        raise RuntimeError(
            f"Could not resolve revision for {repo_id}"
        )

    return info.sha


def prepare_dataset(
    repo_id,
    revision,
    converter,
    output_name,
):
    print(f"\nLoading {repo_id}")
    print(f"revision: {revision}")

    dataset = load_dataset(
        repo_id,
        split="test",
        revision=revision,
        token=HF_TOKEN,
    )

    print(
        f"official test rows: "
        f"{len(dataset)}"
    )

    # Fixed deterministic selection.
    dataset = dataset.shuffle(
        seed=SEED
    )

    selected = []
    rejected = Counter()

    for row in dataset:
        converted, reason = converter(row)

        if converted is None:
            rejected[
                reason or "unknown"
            ] += 1
            continue

        selected.append(converted)

        if len(selected) >= SAMPLE_SIZE:
            break

    if len(selected) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Could only prepare "
            f"{len(selected)} valid rows "
            f"from {repo_id}; "
            f"expected {SAMPLE_SIZE}."
        )

    output_path = (
        OUTPUT_DIR / output_name
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in selected:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    mapped_counts = Counter()
    ignored_counts = Counter()

    for row in selected:
        for entity in row["entities"]:
            mapped_counts[
                entity["type"]
            ] += 1

        for entity in row[
            "ignored_entities"
        ]:
            ignored_counts[
                entity["source_type"]
            ] += 1

    mapped_total = sum(
        mapped_counts.values()
    )
    ignored_total = sum(
        ignored_counts.values()
    )

    stats = {
        "repo_id": repo_id,
        "revision": revision,
        "source_split": "test",
        "seed": SEED,
        "sample_size": len(selected),
        "mapped_entity_count":
            mapped_total,
        "ignored_entity_count":
            ignored_total,
        "mapped_type_counts": dict(
            sorted(
                mapped_counts.items()
            )
        ),
        "top_ignored_source_types":
            dict(
                ignored_counts.most_common(
                    30
                )
            ),
        "rejected_rows_before_selection":
            dict(
                sorted(
                    rejected.items()
                )
            ),
        "output_file":
            str(
                output_path.relative_to(
                    PROJECT_ROOT
                )
            ),
    }

    print(
        f"saved {len(selected)} rows -> "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"mapped entities : "
        f"{mapped_total}"
    )
    print(
        f"ignored entities: "
        f"{ignored_total}"
    )

    if rejected:
        print(
            "rejected before selection: "
            + ", ".join(
                f"{key}={value}"
                for key, value
                in sorted(
                    rejected.items()
                )
            )
        )

    return stats


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "PREPARE EXTERNAL PII BENCHMARKS"
    )
    print("=" * 72)

    if HF_TOKEN:
        print(
            "Hugging Face authentication: enabled"
        )
    else:
        print(
            "Hugging Face authentication: "
            "not found (public download will still work)"
        )

    gretel_revision = resolve_revision(
        GRETEL_REPO
    )
    nemotron_revision = (
        resolve_revision(
            NEMOTRON_REPO
        )
    )

    gretel_stats = prepare_dataset(
        GRETEL_REPO,
        gretel_revision,
        convert_gretel_row,
        "gretel_test_1000.jsonl",
    )

    nemotron_stats = prepare_dataset(
        NEMOTRON_REPO,
        nemotron_revision,
        convert_nemotron_row,
        "nemotron_test_1000.jsonl",
    )

    manifest = {
        "seed": SEED,
        "sample_size_per_dataset":
            SAMPLE_SIZE,
        "evaluation_taxonomy":
            sorted(ALLOWED_TYPES),
        "label_mapping":
            dict(
                sorted(
                    LABEL_MAP.items()
                )
            ),
        "policy": (
            "Only source labels with a clear "
            "equivalent in PrivacyGuard's "
            "OpenPII taxonomy are scored. "
            "Other annotated spans are stored "
            "as ignored_entities."
        ),
        "datasets": {
            "gretel": gretel_stats,
            "nemotron": nemotron_stats,
        },
    }

    manifest_path = (
        OUTPUT_DIR
        / "benchmark_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nmanifest -> "
        f"{manifest_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
