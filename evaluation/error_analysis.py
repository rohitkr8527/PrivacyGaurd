import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_FILE = DATA_DIR / "test.jsonl"
BASELINE_FILE = RESULTS_DIR / "baseline_predictions.jsonl"
FINETUNED_FILE = RESULTS_DIR / "finetuned_predictions.jsonl"

ENTITY_METRICS_FILE = RESULTS_DIR / "entity_metrics.json"
ERROR_ANALYSIS_FILE = RESULTS_DIR / "error_analysis.json"


def load_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}"
                ) from exc
    return rows


def entity_key(entity):
    return (
        int(entity["start"]),
        int(entity["end"]),
        str(entity["type"]).upper(),
    )


def span_key(entity):
    return (
        int(entity["start"]),
        int(entity["end"]),
    )


def safe_div(a, b):
    return a / b if b else 0.0


def calculate_per_entity_metrics(predictions):
    labels = set()

    for row in predictions:
        for entity in row["gold_entities"]:
            labels.add(str(entity["type"]).upper())
        for entity in row["predicted_entities"]:
            labels.add(str(entity["type"]).upper())

    counts = {
        label: {"tp": 0, "fp": 0, "fn": 0}
        for label in sorted(labels)
    }

    micro_tp = micro_fp = micro_fn = 0

    for row in predictions:
        gold = {entity_key(x) for x in row["gold_entities"]}
        predicted = (
            {entity_key(x) for x in row["predicted_entities"]}
            if row.get("valid_json", False)
            else set()
        )

        tp = gold & predicted
        fp = predicted - gold
        fn = gold - predicted

        micro_tp += len(tp)
        micro_fp += len(fp)
        micro_fn += len(fn)

        for _, _, label in tp:
            counts[label]["tp"] += 1
        for _, _, label in fp:
            counts[label]["fp"] += 1
        for _, _, label in fn:
            counts[label]["fn"] += 1

    by_entity = {}

    for label in sorted(counts):
        tp = counts[label]["tp"]
        fp = counts[label]["fp"]
        fn = counts[label]["fn"]

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        by_entity[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "support": tp + fn,
        }

    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )
    macro_f1 = safe_div(
        sum(x["f1"] for x in by_entity.values()),
        len(by_entity),
    )

    return {
        "overall": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "macro_f1": macro_f1,
            "true_positives": micro_tp,
            "false_positives": micro_fp,
            "false_negatives": micro_fn,
        },
        "by_entity": by_entity,
    }


def analyze_errors(predictions, test_by_uid):
    false_negative_by_type = Counter()
    false_positive_by_type = Counter()
    wrong_label_pairs = Counter()

    false_negative_examples = defaultdict(list)
    false_positive_examples = defaultdict(list)
    wrong_label_examples = defaultdict(list)

    invalid_json_count = 0

    for row in predictions:
        uid = str(row["uid"])
        test_row = test_by_uid.get(uid)
        if test_row is None:
            continue

        if not row.get("valid_json", False):
            invalid_json_count += 1

        gold_entities = row["gold_entities"]
        predicted_entities = (
            row["predicted_entities"]
            if row.get("valid_json", False)
            else []
        )

        gold_by_key = {
            entity_key(entity): entity
            for entity in gold_entities
        }
        pred_by_key = {
            entity_key(entity): entity
            for entity in predicted_entities
        }

        gold_keys = set(gold_by_key)
        pred_keys = set(pred_by_key)

        false_negative_keys = gold_keys - pred_keys
        false_positive_keys = pred_keys - gold_keys

        gold_by_span = defaultdict(set)
        pred_by_span = defaultdict(set)

        for entity in gold_entities:
            gold_by_span[span_key(entity)].add(
                str(entity["type"]).upper()
            )

        for entity in predicted_entities:
            pred_by_span[span_key(entity)].add(
                str(entity["type"]).upper()
            )

        for span in set(gold_by_span) & set(pred_by_span):
            for gold_type in gold_by_span[span]:
                for pred_type in pred_by_span[span]:
                    if gold_type == pred_type:
                        continue

                    pair = (gold_type, pred_type)
                    wrong_label_pairs[pair] += 1

                    if len(wrong_label_examples[pair]) < 5:
                        wrong_label_examples[pair].append(
                            {
                                "uid": uid,
                                "text": test_row["text"],
                                "start": span[0],
                                "end": span[1],
                                "value": test_row["text"][span[0]:span[1]],
                                "gold_type": gold_type,
                                "predicted_type": pred_type,
                            }
                        )

        for key in false_negative_keys:
            _, _, label = key
            false_negative_by_type[label] += 1

            if len(false_negative_examples[label]) < 5:
                false_negative_examples[label].append(
                    {
                        "uid": uid,
                        "text": test_row["text"],
                        "entity": gold_by_key[key],
                    }
                )

        for key in false_positive_keys:
            _, _, label = key
            false_positive_by_type[label] += 1

            if len(false_positive_examples[label]) < 5:
                false_positive_examples[label].append(
                    {
                        "uid": uid,
                        "text": test_row["text"],
                        "entity": pred_by_key[key],
                    }
                )

    return {
        "invalid_json_count": invalid_json_count,
        "false_negative_count": sum(false_negative_by_type.values()),
        "false_positive_count": sum(false_positive_by_type.values()),
        "wrong_label_count": sum(wrong_label_pairs.values()),
        "most_missed_entity_types": [
            {"type": label, "count": count}
            for label, count in false_negative_by_type.most_common()
        ],
        "most_over_predicted_entity_types": [
            {"type": label, "count": count}
            for label, count in false_positive_by_type.most_common()
        ],
        "wrong_label_pairs": [
            {
                "gold_type": gold,
                "predicted_type": pred,
                "count": count,
                "examples": wrong_label_examples[(gold, pred)],
            }
            for (gold, pred), count in wrong_label_pairs.most_common()
        ],
        "false_negative_examples": dict(false_negative_examples),
        "false_positive_examples": dict(false_positive_examples),
    }


def print_entity_table(name, metrics):
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)
    print(
        f"{'entity':20s}"
        f"{'precision':>12s}"
        f"{'recall':>12s}"
        f"{'f1':>12s}"
        f"{'support':>10s}"
    )
    print("-" * 78)

    for label, item in metrics["by_entity"].items():
        print(
            f"{label:20s}"
            f"{item['precision']:12.4f}"
            f"{item['recall']:12.4f}"
            f"{item['f1']:12.4f}"
            f"{item['support']:10d}"
        )

    print("-" * 78)
    overall = metrics["overall"]

    print(
        f"{'MICRO':20s}"
        f"{overall['precision']:12.4f}"
        f"{overall['recall']:12.4f}"
        f"{overall['f1']:12.4f}"
    )
    print(
        f"{'MACRO F1':20s}"
        f"{'':>12s}"
        f"{'':>12s}"
        f"{overall['macro_f1']:12.4f}"
    )


def print_error_summary(name, analysis):
    print("\n" + "=" * 78)
    print(f"{name} ERROR SUMMARY")
    print("=" * 78)
    print(f"false negatives : {analysis['false_negative_count']}")
    print(f"false positives : {analysis['false_positive_count']}")
    print(f"wrong labels    : {analysis['wrong_label_count']}")
    print(f"invalid JSON    : {analysis['invalid_json_count']}")

    print("\nMost missed entity types:")
    for item in analysis["most_missed_entity_types"][:10]:
        print(f"  {item['type']:20s} {item['count']}")

    print("\nMost over-predicted entity types:")
    for item in analysis["most_over_predicted_entity_types"][:10]:
        print(f"  {item['type']:20s} {item['count']}")

    print("\nMost common wrong labels:")
    for item in analysis["wrong_label_pairs"][:10]:
        print(
            f"  {item['gold_type']:18s} -> "
            f"{item['predicted_type']:18s} "
            f"{item['count']}"
        )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_rows = load_jsonl(TEST_FILE)
    test_by_uid = {
        str(row["uid"]): row
        for row in test_rows
    }

    baseline_predictions = load_jsonl(BASELINE_FILE)
    finetuned_predictions = load_jsonl(FINETUNED_FILE)

    baseline_metrics = calculate_per_entity_metrics(
        baseline_predictions
    )
    finetuned_metrics = calculate_per_entity_metrics(
        finetuned_predictions
    )

    baseline_errors = analyze_errors(
        baseline_predictions,
        test_by_uid,
    )
    finetuned_errors = analyze_errors(
        finetuned_predictions,
        test_by_uid,
    )

    with ENTITY_METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "baseline": baseline_metrics,
                "finetuned": finetuned_metrics,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with ERROR_ANALYSIS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "baseline": baseline_errors,
                "finetuned": finetuned_errors,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_entity_table(
        "BASELINE PER-ENTITY METRICS",
        baseline_metrics,
    )
    print_entity_table(
        "FINE-TUNED PER-ENTITY METRICS",
        finetuned_metrics,
    )
    print_error_summary("BASELINE", baseline_errors)
    print_error_summary("FINE-TUNED", finetuned_errors)

    print(f"\nEntity metrics -> {ENTITY_METRICS_FILE}")
    print(f"Error analysis -> {ERROR_ANALYSIS_FILE}")


if __name__ == "__main__":
    main()
