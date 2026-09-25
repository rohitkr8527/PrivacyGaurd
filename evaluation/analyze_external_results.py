import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EXTERNAL_DIR = RESULTS_DIR / "external"

BASELINE_METRICS_FILE = RESULTS_DIR / "baseline_metrics.json"
FINETUNED_METRICS_FILE = RESULTS_DIR / "finetuned_metrics.json"
EXTERNAL_METRICS_FILE = EXTERNAL_DIR / "external_benchmark_metrics.json"

SUMMARY_FILE = EXTERNAL_DIR / "generalization_summary.json"
ERROR_FILE = EXTERNAL_DIR / "external_error_analysis.json"


CORE_METRICS = (
    "precision",
    "recall",
    "f1",
    "f2",
    "pii_leakage_rate",
    "over_redaction_rate",
    "json_validity",
)


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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
                    f"Invalid JSON in {path} line {line_number}"
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


def overlaps_ignored(entity, ignored_entities):
    start = int(entity["start"])
    end = int(entity["end"])

    for ignored in ignored_entities:
        ignored_start = int(ignored["start"])
        ignored_end = int(ignored["end"])

        if start < ignored_end and end > ignored_start:
            return True

    return False


def extract_metric_block(payload):
    if "metrics" in payload:
        return payload["metrics"]
    return payload


def delta_block(base, finetuned):
    return {
        metric: finetuned[metric] - base[metric]
        for metric in CORE_METRICS
        if metric in base and metric in finetuned
    }


def analyze_prediction_file(path: Path):
    rows = load_jsonl(path)

    fn_by_type = Counter()
    fp_by_type = Counter()
    wrong_label_pairs = Counter()

    wrong_label_examples = defaultdict(list)
    fn_examples = defaultdict(list)
    fp_examples = defaultdict(list)

    invalid_json = 0
    ignored_prediction_count = 0

    for row in rows:
        if not row.get("valid_json", False):
            invalid_json += 1

        gold_entities = row["gold_entities"]

        predicted_entities = (
            row["predicted_entities"]
            if row.get("valid_json", False)
            else []
        )

        scored_predictions = []

        for entity in predicted_entities:
            if overlaps_ignored(
                entity,
                row.get("ignored_entities", []),
            ):
                ignored_prediction_count += 1
                continue

            scored_predictions.append(entity)

        gold_by_key = {
            entity_key(entity): entity
            for entity in gold_entities
        }

        pred_by_key = {
            entity_key(entity): entity
            for entity in scored_predictions
        }

        gold_keys = set(gold_by_key)
        pred_keys = set(pred_by_key)

        fn_keys = gold_keys - pred_keys
        fp_keys = pred_keys - gold_keys

        for key in fn_keys:
            _, _, label = key
            fn_by_type[label] += 1

            if len(fn_examples[label]) < 5:
                fn_examples[label].append(
                    {
                        "uid": row["uid"],
                        "entity": gold_by_key[key],
                        "text": row["text"],
                    }
                )

        for key in fp_keys:
            _, _, label = key
            fp_by_type[label] += 1

            if len(fp_examples[label]) < 5:
                fp_examples[label].append(
                    {
                        "uid": row["uid"],
                        "entity": pred_by_key[key],
                        "text": row["text"],
                    }
                )

        gold_by_span = defaultdict(set)
        pred_by_span = defaultdict(set)

        for entity in gold_entities:
            gold_by_span[span_key(entity)].add(
                str(entity["type"]).upper()
            )

        for entity in scored_predictions:
            pred_by_span[span_key(entity)].add(
                str(entity["type"]).upper()
            )

        for span in set(gold_by_span) & set(pred_by_span):
            for gold_type in gold_by_span[span]:
                for predicted_type in pred_by_span[span]:
                    if gold_type == predicted_type:
                        continue

                    pair = (gold_type, predicted_type)
                    wrong_label_pairs[pair] += 1

                    if len(wrong_label_examples[pair]) < 5:
                        wrong_label_examples[pair].append(
                            {
                                "uid": row["uid"],
                                "value": row["text"][
                                    span[0]:span[1]
                                ],
                                "gold_type": gold_type,
                                "predicted_type": predicted_type,
                                "text": row["text"],
                            }
                        )

    return {
        "sample_count": len(rows),
        "invalid_json_count": invalid_json,
        "ignored_prediction_count": ignored_prediction_count,
        "false_negative_count": sum(fn_by_type.values()),
        "false_positive_count": sum(fp_by_type.values()),
        "wrong_label_count": sum(wrong_label_pairs.values()),
        "most_missed_entity_types": [
            {"type": label, "count": count}
            for label, count in fn_by_type.most_common()
        ],
        "most_over_predicted_entity_types": [
            {"type": label, "count": count}
            for label, count in fp_by_type.most_common()
        ],
        "wrong_label_pairs": [
            {
                "gold_type": gold,
                "predicted_type": pred,
                "count": count,
                "examples": wrong_label_examples[(gold, pred)],
            }
            for (gold, pred), count
            in wrong_label_pairs.most_common()
        ],
        "false_negative_examples": dict(fn_examples),
        "false_positive_examples": dict(fp_examples),
    }


def core_only(metrics):
    return {
        key: metrics[key]
        for key in CORE_METRICS
        if key in metrics
    }


def print_comparison(dataset, base, finetuned):
    print("\n" + "=" * 86)
    print(dataset.upper())
    print("=" * 86)
    print(
        f"{'metric':24s}"
        f"{'base':>14s}"
        f"{'fine-tuned':>14s}"
        f"{'delta':>14s}"
    )
    print("-" * 86)

    for metric in CORE_METRICS:
        if metric not in base or metric not in finetuned:
            continue

        delta = finetuned[metric] - base[metric]

        print(
            f"{metric:24s}"
            f"{base[metric]:14.4f}"
            f"{finetuned[metric]:14.4f}"
            f"{delta:+14.4f}"
        )


def print_weak_entities(dataset, metrics):
    by_entity = metrics.get("by_entity", {})

    if not by_entity:
        return

    # Only rank entity types that actually occur in the benchmark.
    # Labels with support=0 are not measurable and should not appear
    # as "weakest" classes merely because their computed F1 is 0.
    measurable = [
        (label, item)
        for label, item in by_entity.items()
        if int(item.get("support", 0)) > 0
    ]

    ordered = sorted(
        measurable,
        key=lambda item: (
            item[1]["f1"],
            item[0],
        ),
    )

    print(f"\nLowest fine-tuned {dataset} entity F1 (support > 0):")

    for label, item in ordered[:8]:
        print(
            f"  {label:22s} "
            f"P={item['precision']:.4f} "
            f"R={item['recall']:.4f} "
            f"F1={item['f1']:.4f} "
            f"support={item['support']}"
        )


def main():
    baseline_payload = load_json(
        BASELINE_METRICS_FILE
    )
    finetuned_payload = load_json(
        FINETUNED_METRICS_FILE
    )
    external_payload = load_json(
        EXTERNAL_METRICS_FILE
    )

    baseline_openpii = extract_metric_block(
        baseline_payload
    )
    finetuned_openpii = extract_metric_block(
        finetuned_payload
    )

    summary = {
        "openpii": {
            "base": core_only(baseline_openpii),
            "finetuned": core_only(finetuned_openpii),
            "delta": delta_block(
                baseline_openpii,
                finetuned_openpii,
            ),
        },
        "external": {},
        "generalization_gap_vs_openpii_finetuned": {},
    }

    errors = {}

    print_comparison(
        "OpenPII in-domain",
        baseline_openpii,
        finetuned_openpii,
    )

    for dataset_name in ("gretel", "nemotron"):
        dataset_payload = (
            external_payload["datasets"][dataset_name]
        )

        base_metrics = dataset_payload["base"]["metrics"]
        finetuned_metrics = dataset_payload[
            "finetuned"
        ]["metrics"]

        summary["external"][dataset_name] = {
            "base": core_only(base_metrics),
            "finetuned": core_only(finetuned_metrics),
            "delta": delta_block(
                base_metrics,
                finetuned_metrics,
            ),
        }

        summary[
            "generalization_gap_vs_openpii_finetuned"
        ][dataset_name] = {
            "f1": (
                finetuned_metrics["f1"]
                - finetuned_openpii["f1"]
            ),
            "recall": (
                finetuned_metrics["recall"]
                - finetuned_openpii["recall"]
            ),
            "f2": (
                finetuned_metrics["f2"]
                - finetuned_openpii["f2"]
            ),
        }

        print_comparison(
            f"{dataset_name} external",
            base_metrics,
            finetuned_metrics,
        )

        print_weak_entities(
            dataset_name,
            finetuned_metrics,
        )

        errors[dataset_name] = {}

        for model_name in ("base", "finetuned"):
            prediction_path = (
                EXTERNAL_DIR
                / (
                    f"{dataset_name}_"
                    f"{model_name}_predictions.jsonl"
                )
            )

            errors[dataset_name][
                model_name
            ] = analyze_prediction_file(
                prediction_path
            )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with ERROR_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            errors,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 86)
    print("GENERALIZATION SUMMARY")
    print("=" * 86)

    openpii_f1 = finetuned_openpii["f1"]

    print(
        f"OpenPII fine-tuned F1 : {openpii_f1:.4f}"
    )

    for dataset_name in ("gretel", "nemotron"):
        ft = summary["external"][
            dataset_name
        ]["finetuned"]

        delta = summary["external"][
            dataset_name
        ]["delta"]

        gap = summary[
            "generalization_gap_vs_openpii_finetuned"
        ][dataset_name]

        print(
            f"{dataset_name.capitalize():10s} "
            f"fine-tuned F1={ft['f1']:.4f} "
            f"(vs base {delta['f1']:+.4f}), "
            f"Recall={ft['recall']:.4f} "
            f"(vs base {delta['recall']:+.4f}), "
            f"F1 gap vs OpenPII={gap['f1']:+.4f}"
        )

    print(
        f"\nSummary -> {SUMMARY_FILE}"
    )
    print(
        f"Errors  -> {ERROR_FILE}"
    )


if __name__ == "__main__":
    main()
