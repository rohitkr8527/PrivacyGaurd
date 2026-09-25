import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

BASELINE_FILE = RESULTS_DIR / "baseline_metrics.json"
FINETUNED_FILE = RESULTS_DIR / "finetuned_metrics.json"
OUTPUT_FILE = RESULTS_DIR / "baseline_vs_finetuned.json"

METRICS = [
    "precision",
    "recall",
    "f1",
    "f2",
    "pii_leakage_rate",
    "over_redaction_rate",
    "json_validity",
]


def load_metrics(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid metrics file: {path}")

    return metrics


def main():
    baseline = load_metrics(BASELINE_FILE)
    finetuned = load_metrics(FINETUNED_FILE)

    comparison = {}

    print("\n" + "=" * 68)
    print("BASELINE VS FINE-TUNED")
    print("=" * 68)
    print(
        f"{'metric':24s}"
        f"{'baseline':>12s}"
        f"{'fine-tuned':>14s}"
        f"{'delta':>12s}"
    )
    print("-" * 68)

    for metric in METRICS:
        b = float(baseline[metric])
        f = float(finetuned[metric])
        delta = f - b

        comparison[metric] = {
            "baseline": b,
            "finetuned": f,
            "delta": delta,
        }

        print(
            f"{metric:24s}"
            f"{b:12.4f}"
            f"{f:14.4f}"
            f"{delta:+12.4f}"
        )

    print("=" * 68)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
