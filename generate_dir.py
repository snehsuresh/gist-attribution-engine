import os

PROJECT_ROOT = "."

structure = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "data/raw/news/",
    "data/raw/logs/",
    "data/processed/duckdb/",
    "ingestion/news_etl.py",
    "ingestion/userlog_generator.py",
    "ingestion/normalize.py",
    "embedding/embedder.py",
    "embedding/vector_store.py",
    "embedding/config.json",
    "attribution/ablation.py",
    "attribution/score.py",
    "explanation/summarizer.py",
    "explanation/prompt_templates.py",
    "frontend/app.py",
    "frontend/ui/components.py",
    "frontend/logs/queries.duckdb",
    "utils/helpers.py",
    "tests/test_ingestion.py",
    "tests/test_embedding.py",
]


def create_structure(root, paths):
    for path in paths:
        full_path = os.path.join(root, path)
        if path.endswith("/"):
            os.makedirs(full_path, exist_ok=True)
        elif "." in os.path.basename(path):  # likely a file
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                pass  # create empty file


if __name__ == "__main__":
    create_structure(PROJECT_ROOT, structure)
    print(f"✅ Project scaffold created in: {PROJECT_ROOT}")
