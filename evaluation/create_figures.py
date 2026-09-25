import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BASELINE_METRICS_FILE = RESULTS_DIR / "baseline_metrics.json"
FINETUNED_METRICS_FILE = RESULTS_DIR / "finetuned_metrics.json"
ENTITY_METRICS_FILE = RESULTS_DIR / "entity_metrics.json"
GENERALIZATION_FILE = (
    RESULTS_DIR
    / "external"
    / "generalization_summary.json"
)


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_block(payload):
    return payload.get("metrics", payload)


def add_value_labels(ax, bars, decimals=3):
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_openpii_comparison(base, finetuned):
    metrics = [
        "precision",
        "recall",
        "f1",
        "f2",
    ]

    labels = [
        "Precision",
        "Recall",
        "F1",
        "F2",
    ]

    base_values = [
        base[key]
        for key in metrics
    ]

    finetuned_values = [
        finetuned[key]
        for key in metrics
    ]

    x = list(range(len(labels)))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    base_bars = ax.bar(
        [value - width / 2 for value in x],
        base_values,
        width,
        label="Base Qwen3-4B",
    )

    finetuned_bars = ax.bar(
        [value + width / 2 for value in x],
        finetuned_values,
        width,
        label="Fine-tuned Qwen3-4B",
    )

    ax.set_title(
        "OpenPII: Base vs Fine-tuned Performance"
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    add_value_labels(ax, base_bars)
    add_value_labels(ax, finetuned_bars)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "openpii_model_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_privacy_risk_comparison(
    base,
    finetuned,
):
    metrics = [
        "pii_leakage_rate",
        "over_redaction_rate",
    ]

    labels = [
        "PII Leakage",
        "Over-redaction",
    ]

    base_values = [
        base[key]
        for key in metrics
    ]

    finetuned_values = [
        finetuned[key]
        for key in metrics
    ]

    x = list(range(len(labels)))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    base_bars = ax.bar(
        [value - width / 2 for value in x],
        base_values,
        width,
        label="Base Qwen3-4B",
    )

    finetuned_bars = ax.bar(
        [value + width / 2 for value in x],
        finetuned_values,
        width,
        label="Fine-tuned Qwen3-4B",
    )

    ax.set_title(
        "OpenPII: Privacy Error Rates"
    )
    ax.set_ylabel("Rate")
    ax.set_ylim(
        0,
        max(
            max(base_values),
            max(finetuned_values),
        )
        * 1.25,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    add_value_labels(ax, base_bars)
    add_value_labels(ax, finetuned_bars)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "openpii_privacy_error_rates.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_generalization_f1(
    base_openpii,
    finetuned_openpii,
    generalization,
):
    datasets = [
        "OpenPII",
        "Gretel",
        "Nemotron",
    ]

    base_values = [
        base_openpii["f1"],
        generalization[
            "external"
        ]["gretel"]["base"]["f1"],
        generalization[
            "external"
        ]["nemotron"]["base"]["f1"],
    ]

    finetuned_values = [
        finetuned_openpii["f1"],
        generalization[
            "external"
        ]["gretel"]["finetuned"]["f1"],
        generalization[
            "external"
        ]["nemotron"]["finetuned"]["f1"],
    ]

    x = list(range(len(datasets)))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    base_bars = ax.bar(
        [value - width / 2 for value in x],
        base_values,
        width,
        label="Base Qwen3-4B",
    )

    finetuned_bars = ax.bar(
        [value + width / 2 for value in x],
        finetuned_values,
        width,
        label="Fine-tuned Qwen3-4B",
    )

    ax.set_title(
        "Generalization: F1 Across Benchmarks"
    )
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    add_value_labels(ax, base_bars)
    add_value_labels(ax, finetuned_bars)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "generalization_f1.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_generalization_recall(
    base_openpii,
    finetuned_openpii,
    generalization,
):
    datasets = [
        "OpenPII",
        "Gretel",
        "Nemotron",
    ]

    base_values = [
        base_openpii["recall"],
        generalization[
            "external"
        ]["gretel"]["base"]["recall"],
        generalization[
            "external"
        ]["nemotron"]["base"]["recall"],
    ]

    finetuned_values = [
        finetuned_openpii["recall"],
        generalization[
            "external"
        ]["gretel"]["finetuned"]["recall"],
        generalization[
            "external"
        ]["nemotron"]["finetuned"]["recall"],
    ]

    x = list(range(len(datasets)))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    base_bars = ax.bar(
        [value - width / 2 for value in x],
        base_values,
        width,
        label="Base Qwen3-4B",
    )

    finetuned_bars = ax.bar(
        [value + width / 2 for value in x],
        finetuned_values,
        width,
        label="Fine-tuned Qwen3-4B",
    )

    ax.set_title(
        "Generalization: Recall Across Benchmarks"
    )
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    add_value_labels(ax, base_bars)
    add_value_labels(ax, finetuned_bars)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "generalization_recall.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_entity_f1(entity_metrics):
    by_entity = (
        entity_metrics[
            "finetuned"
        ]["by_entity"]
    )

    ordered = sorted(
        [
            (
                label,
                values["f1"],
            )
            for label, values
            in by_entity.items()
            if values.get("support", 0) > 0
        ],
        key=lambda item: item[1],
    )

    labels = [
        item[0]
        for item in ordered
    ]

    values = [
        item[1]
        for item in ordered
    ]

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    bars = ax.barh(
        labels,
        values,
    )

    ax.set_title(
        "Fine-tuned OpenPII F1 by Entity Type"
    )
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1.05)

    for bar in bars:
        width = bar.get_width()

        ax.text(
            width,
            bar.get_y()
            + bar.get_height() / 2,
            f" {width:.3f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "finetuned_entity_f1.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_payload = load_json(
        BASELINE_METRICS_FILE
    )

    finetuned_payload = load_json(
        FINETUNED_METRICS_FILE
    )

    entity_metrics = load_json(
        ENTITY_METRICS_FILE
    )

    generalization = load_json(
        GENERALIZATION_FILE
    )

    baseline = metric_block(
        baseline_payload
    )

    finetuned = metric_block(
        finetuned_payload
    )

    save_openpii_comparison(
        baseline,
        finetuned,
    )

    save_privacy_risk_comparison(
        baseline,
        finetuned,
    )

    save_generalization_f1(
        baseline,
        finetuned,
        generalization,
    )

    save_generalization_recall(
        baseline,
        finetuned,
        generalization,
    )

    save_entity_f1(
        entity_metrics
    )

    print("Generated figures:")
    for path in sorted(
        FIGURES_DIR.glob("*.png")
    ):
        print(
            f"  {path.relative_to(PROJECT_ROOT)}"
        )


if __name__ == "__main__":
    main()
