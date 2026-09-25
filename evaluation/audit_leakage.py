import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "validation.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

BASELINE_PREDICTIONS = RESULTS_DIR / "baseline_predictions.jsonl"
FINETUNED_PREDICTIONS = RESULTS_DIR / "finetuned_predictions.jsonl"
AUDIT_OUTPUT = RESULTS_DIR / "leakage_audit.json"


def load_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}"
                ) from exc
    return rows


def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def template_signature(row):
    """
    Replace every gold PII span with its entity type.
    This catches rows that use the same synthetic sentence/template
    with different names, dates, phone numbers, etc.
    """
    text = row["text"]
    entities = sorted(
        row["entities"],
        key=lambda x: (int(x["start"]), int(x["end"])),
        reverse=True,
    )

    for entity in entities:
        start = int(entity["start"])
        end = int(entity["end"])
        label = str(entity["type"]).upper()
        text = text[:start] + f"<{label}>" + text[end:]

    return normalize_text(text)


def entity_value_keys(rows):
    keys = set()
    for row in rows:
        for entity in row["entities"]:
            keys.add(
                (
                    str(entity["type"]).upper(),
                    normalize_text(str(entity["text"])),
                )
            )
    return keys


def entity_set(entities):
    return {
        (
            int(entity["start"]),
            int(entity["end"]),
            str(entity["type"]).upper(),
        )
        for entity in entities
        if {"start", "end", "type"} <= set(entity)
    }


def micro_metrics(items):
    tp = fp = fn = 0

    for item in items:
        gold = entity_set(item["gold_entities"])
        pred = entity_set(item["predicted_entities"])

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "samples": len(items),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_metrics(name, items):
    m = micro_metrics(items)
    print(
        f"{name:28s} "
        f"n={m['samples']:4d}  "
        f"P={m['precision']:.4f}  "
        f"R={m['recall']:.4f}  "
        f"F1={m['f1']:.4f}"
    )


def audit_predictions(path, test_by_uid, train_templates, train_exact):
    if not path.is_file():
        return

    predictions = load_jsonl(path)

    seen_template = []
    unseen_template = []
    exact_seen = []
    exact_unseen = []

    missing_uid = 0

    for pred in predictions:
        uid = str(pred["uid"])
        row = test_by_uid.get(uid)

        if row is None:
            missing_uid += 1
            continue

        record = {
            "gold_entities": pred["gold_entities"],
            "predicted_entities": (
                pred["predicted_entities"]
                if pred.get("valid_json", False)
                else []
            ),
        }

        if template_signature(row) in train_templates:
            seen_template.append(record)
        else:
            unseen_template.append(record)

        if row["text"] in train_exact:
            exact_seen.append(record)
        else:
            exact_unseen.append(record)

    print(f"\nPrediction audit: {path.name}")
    print("-" * 72)
    print_metrics("all test rows", seen_template + unseen_template)
    print_metrics("train-template seen", seen_template)
    print_metrics("train-template unseen", unseen_template)
    print_metrics("exact train-text seen", exact_seen)
    print_metrics("exact train-text unseen", exact_unseen)

    if missing_uid:
        print(f"WARNING: {missing_uid} prediction UIDs not found in test.jsonl")


def main():
    train = load_jsonl(TRAIN_FILE)
    val = load_jsonl(VAL_FILE)
    test = load_jsonl(TEST_FILE)

    print("=" * 72)
    print("PRIVACYGUARD DATA LEAKAGE / SPLIT AUDIT")
    print("=" * 72)

    print(f"train rows      : {len(train)}")
    print(f"validation rows : {len(val)}")
    print(f"test rows       : {len(test)}")

    train_uids = {str(x["uid"]) for x in train}
    val_uids = {str(x["uid"]) for x in val}
    test_uids = {str(x["uid"]) for x in test}

    uid_train_val = len(train_uids & val_uids)
    uid_train_test = len(train_uids & test_uids)
    uid_val_test = len(val_uids & test_uids)

    print("\nUID overlap")
    print("-" * 72)
    print(f"train ∩ validation : {uid_train_val}")
    print(f"train ∩ test       : {uid_train_test}")
    print(f"validation ∩ test  : {uid_val_test}")

    train_exact = {x["text"] for x in train}
    val_exact = {x["text"] for x in val}

    exact_train_test = len(train_exact & {x["text"] for x in test})
    exact_val_test = len(val_exact & {x["text"] for x in test})
    exact_seen_rows = sum(x["text"] in train_exact for x in test)

    print("\nExact text overlap")
    print("-" * 72)
    print(f"train ∩ test       : {exact_train_test}")
    print(f"validation ∩ test  : {exact_val_test}")
    print(
        "test rows exactly seen in train: "
        f"{exact_seen_rows}/{len(test)} "
        f"({exact_seen_rows/len(test):.2%})"
    )

    train_norm = {normalize_text(x["text"]) for x in train}
    norm_seen_count = sum(
        normalize_text(x["text"]) in train_norm
        for x in test
    )

    print("\nNormalized text overlap")
    print("-" * 72)
    print(
        f"test rows normalized-seen in train: "
        f"{norm_seen_count}/{len(test)} "
        f"({norm_seen_count/len(test):.2%})"
    )

    print("\nSynthetic template overlap")
    print("-" * 72)

    train_templates = {template_signature(x) for x in train}
    val_templates = {template_signature(x) for x in val}
    test_templates = {template_signature(x) for x in test}

    test_template_seen = sum(
        template_signature(x) in train_templates
        for x in test
    )

    unique_template_intersection = len(
        train_templates & test_templates
    )
    val_test_template_intersection = len(
        val_templates & test_templates
    )

    print(f"unique train templates : {len(train_templates)}")
    print(f"unique test templates  : {len(test_templates)}")
    print(
        f"test rows using a train template: "
        f"{test_template_seen}/{len(test)} "
        f"({test_template_seen/len(test):.2%})"
    )
    print(
        f"unique template intersection: "
        f"{unique_template_intersection}"
    )
    print(
        f"validation/test template intersection: "
        f"{val_test_template_intersection}"
    )

    train_entity_values = entity_value_keys(train)
    total_test_entities = 0
    seen_test_entities = 0
    seen_by_type = Counter()
    total_by_type = Counter()

    for row in test:
        for entity in row["entities"]:
            label = str(entity["type"]).upper()
            key = (
                label,
                normalize_text(str(entity["text"])),
            )

            total_test_entities += 1
            total_by_type[label] += 1

            if key in train_entity_values:
                seen_test_entities += 1
                seen_by_type[label] += 1

    entity_overlap_percentage = (
        seen_test_entities / total_test_entities * 100
        if total_test_entities
        else 0.0
    )

    print("\nEntity-value overlap")
    print("-" * 72)
    print(
        f"test entity values seen in train: "
        f"{seen_test_entities}/{total_test_entities} "
        f"({entity_overlap_percentage:.2f}%)"
    )

    entity_overlap_by_type = {}

    for label in sorted(total_by_type):
        total = total_by_type[label]
        seen = seen_by_type[label]

        entity_overlap_by_type[label] = {
            "seen": seen,
            "total": total,
            "percentage": round(
                seen / total * 100 if total else 0.0,
                2,
            ),
        }

        print(
            f"{label:20s} {seen:5d}/{total:5d} "
            f"({seen/total:.2%})"
        )

    test_by_uid = {
        str(row["uid"]): row
        for row in test
    }

    audit_predictions(
        BASELINE_PREDICTIONS,
        test_by_uid,
        train_templates,
        train_exact,
    )

    audit_predictions(
        FINETUNED_PREDICTIONS,
        test_by_uid,
        train_templates,
        train_exact,
    )

    finetuned_f1_all = None
    finetuned_f1_unseen_templates = None

    if FINETUNED_PREDICTIONS.is_file():
        predictions = load_jsonl(
            FINETUNED_PREDICTIONS
        )

        all_items = []
        unseen_items = []

        for pred in predictions:
            uid = str(pred["uid"])
            row = test_by_uid.get(uid)

            if row is None:
                continue

            item = {
                "gold_entities":
                    pred["gold_entities"],
                "predicted_entities":
                    pred["predicted_entities"]
                    if pred.get(
                        "valid_json",
                        False,
                    )
                    else [],
            }

            all_items.append(item)

            if (
                template_signature(row)
                not in train_templates
            ):
                unseen_items.append(item)

        finetuned_f1_all = (
            micro_metrics(all_items)["f1"]
            if all_items
            else None
        )

        finetuned_f1_unseen_templates = (
            micro_metrics(unseen_items)["f1"]
            if unseen_items
            else None
        )

    meaningful_leakage = any(
        [
            uid_train_test > 0,
            exact_train_test > 0,
            norm_seen_count > 0,
            test_template_seen / len(test) > 0.05,
        ]
    )

    conclusion = (
        "Potential train-test leakage detected."
        if meaningful_leakage
        else "No meaningful train-test leakage detected."
    )

    audit_summary = {
        "dataset_sizes": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
        },
        "uid_overlap": {
            "train_validation":
                uid_train_val,
            "train_test":
                uid_train_test,
            "validation_test":
                uid_val_test,
        },
        "exact_text_overlap": {
            "train_test":
                exact_train_test,
            "validation_test":
                exact_val_test,
            "test_rows_seen_in_train":
                exact_seen_rows,
        },
        "normalized_text_overlap_train_test": {
            "count":
                norm_seen_count,
            "total_test_samples":
                len(test),
            "percentage":
                round(
                    norm_seen_count / len(test) * 100,
                    4,
                ),
        },
        "template_overlap_train_test": {
            "count":
                test_template_seen,
            "total_test_samples":
                len(test),
            "percentage":
                round(
                    test_template_seen / len(test) * 100,
                    4,
                ),
            "unique_template_intersection":
                unique_template_intersection,
        },
        "entity_value_overlap": {
            "seen":
                seen_test_entities,
            "total":
                total_test_entities,
            "percentage":
                round(
                    entity_overlap_percentage,
                    2,
                ),
            "by_type":
                entity_overlap_by_type,
        },
        "finetuned_f1_all":
            (
                round(
                    finetuned_f1_all,
                    4,
                )
                if finetuned_f1_all
                is not None
                else None
            ),
        "finetuned_f1_unseen_templates":
            (
                round(
                    finetuned_f1_unseen_templates,
                    4,
                )
                if finetuned_f1_unseen_templates
                is not None
                else None
            ),
        "conclusion":
            conclusion,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print(
        "1. UID overlap > 0 = direct split bug.\n"
        "2. Exact text overlap > 0 = direct content leakage.\n"
        "3. High template overlap means train/test contain the same synthetic\n"
        "   sentence structures with different PII values.\n"
        "4. If fine-tuned F1 is much higher on seen templates than unseen\n"
        "   templates, the 3K score is strongly helped by template leakage.\n"
        "5. Entity-value overlap is useful context, but repeated names/cities\n"
        "   alone do not prove leakage."
    )

    print(
        f"\nAudit summary -> {AUDIT_OUTPUT}"
    )


if __name__ == "__main__":
    main()
