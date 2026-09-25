import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl


# Set better default styling
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10

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

# Color scheme
BASE_COLOR = '#e74c3c'      # Red for base model
FINETUNED_COLOR = '#3498db'  # Blue for fine-tuned model
ENTITY_COLOR = '#2ecc71'     # Green for entity bars


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_block(payload):
    return payload.get("metrics", payload)


def add_value_labels(ax, bars, decimals=3, fontsize=9):
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,  # Small offset above bar
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight='bold',
        )


def save_openpii_comparison(base, finetuned):
    """Create comparison chart for OpenPII metrics."""
    metrics = [
        "precision",
        "recall",
        "f1",
        "f2",
    ]

    labels = [
        "Precision",
        "Recall",
        "F1 Score",
        "F2 Score",
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
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )
    
    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    base_bars = ax.bar(
        [value - width / 2 for value in x],
        base_values,
        width,
        label="Base Qwen3-4B",
        color=BASE_COLOR,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    finetuned_bars = ax.bar(
        [value + width / 2 for value in x],
        finetuned_values,
        width,
        label="Fine-tuned Qwen3-4B",
        color=FINETUNED_COLOR,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    ax.set_title(
        "OpenPII: Base vs Fine-tuned Performance",
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', framealpha=0.9)

    add_value_labels(ax, base_bars, fontsize=8)
    add_value_labels(ax, finetuned_bars, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "openpii_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_privacy_risk_comparison(
    base,
    finetuned,
):
    """Create comparison chart for privacy risk metrics."""
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
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )
    
    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    base_bars = ax.bar(
        [value - width / 2 for value in x],
        base_values,
        width,
        label="Base Qwen3-4B",
        color=BASE_COLOR,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    finetuned_bars = ax.bar(
        [value + width / 2 for value in x],
        finetuned_values,
        width,
        label="Fine-tuned Qwen3-4B",
        color=FINETUNED_COLOR,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    ax.set_title(
        "OpenPII: Privacy Error Rates (Lower is Better)",
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_ylabel("Error Rate", fontsize=12, fontweight='bold')
    ax.set_ylim(
        0,
        max(
            max(base_values),
            max(finetuned_values),
        )
        * 1.3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', framealpha=0.9)

    add_value_labels(ax, base_bars, fontsize=8)
    add_value_labels(ax, finetuned_bars, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "openpii_privacy_error_rates.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_generalization_f1(
    base_openpii,
    finetuned_openpii,
    generalization,
):
    """Create F1 comparison across benchmarks."""
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
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )
    
    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    base_bars = ax.bar(
        [value - width / 2 for value in x],
        base_values,
        width,
        label="Base Qwen3-4B",
        color=BASE_COLOR,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    finetuned_bars = ax.bar(
        [value + width / 2 for value in x],
        finetuned_values,
        width,
        label="Fine-tuned Qwen3-4B",
        color=FINETUNED_COLOR,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    ax.set_title(
        "Generalization: F1 Score Across Benchmarks",
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc='upper right', framealpha=0.9)

    add_value_labels(ax, base_bars, fontsize=8)
    add_value_labels(ax, finetuned_bars, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "generalization_f1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_generalization_recall(
    base_openpii,
    finetuned_openpii,
    generalization,
):
    """Create recall comparison across benchmarks."""
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
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )
    
    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    base_bars = ax.bar(
        [value - width / 2 for value in x],
        base_values,
        width,
        label="Base Qwen3-4B",
        color=BASE_COLOR,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    finetuned_bars = ax.bar(
        [value + width / 2 for value in x],
        finetuned_values,
        width,
        label="Fine-tuned Qwen3-4B",
        color=FINETUNED_COLOR,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    ax.set_title(
        "Generalization: Recall Across Benchmarks",
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_ylabel("Recall", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc='upper right', framealpha=0.9)

    add_value_labels(ax, base_bars, fontsize=8)
    add_value_labels(ax, finetuned_bars, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "generalization_recall.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_entity_f1(entity_metrics):
    """Create horizontal bar chart for entity-level F1 scores."""
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
        figsize=(10, 9)
    )
    
    # Add grid for easier reading
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    bars = ax.barh(
        labels,
        values,
        color=ENTITY_COLOR,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    ax.set_title(
        "Fine-tuned Model: F1 Score by Entity Type",
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_xlabel("F1 Score", fontsize=12, fontweight='bold')
    
    # Set x-axis to start from 0.6 for better clarity
    min_val = min(values) if values else 0.6
    ax.set_xlim(max(0.6, min_val - 0.05), 1.05)
    
    # Improve y-axis labels
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            fontweight='bold',
        )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "finetuned_entity_f1.png",
        dpi=300,
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
