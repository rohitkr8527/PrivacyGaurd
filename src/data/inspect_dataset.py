import os
import sys
from pathlib import Path

# Configure UTF-8 for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Load .env if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with env_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

from datasets import load_dataset

DATASET_NAME = "ai4privacy/pii-masking-openpii-1m"


def main():
    print(f"Loading {DATASET_NAME}...")

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    print("\nColumns:")
    first_example = next(iter(dataset))
    for column in first_example.keys():
        print(f"- {column}")

    print("\nFirst example:")
    for key, value in first_example.items():
        print(f"\n{key}:")
        print(value)


if __name__ == "__main__":
    main()