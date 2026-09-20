from pathlib import Path


PROJECT_ROOT = Path(__file__).parent

DIRECTORIES = [
    "data/raw",
    "data/processed",
    "src/data",
    "training",
    "evaluation",
    "app",
    "results/figures",
]

FILES = [
    "README.md",
    ".gitignore",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "results/figures/.gitkeep",
]


def create_directories():
    for directory in DIRECTORIES:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path.relative_to(PROJECT_ROOT)}")


def create_files():
    for file in FILES:
        path = PROJECT_ROOT / file

        if not path.exists():
            path.touch()
            print(f"Created: {path.relative_to(PROJECT_ROOT)}")


def main():
    print("Setting up PrivacyGuard...\n")

    create_directories()
    create_files()

    print("\nPrivacyGuard structure created successfully.")


if __name__ == "__main__":
    main()