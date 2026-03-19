from __future__ import annotations

import re


def slugify(condition: str) -> str:
    slug = condition.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def build_paths(condition: str) -> dict[str, str]:
    prefix = slugify(condition)
    return {
        "raw":               f"data/{prefix}_raw_examples.txt",
        "train":             f"data/{prefix}_train.txt",
        "test":              f"data/{prefix}_test.txt",
        "patterns":          f"results/{prefix}_patterns.json",
        "regex_patterns":    f"results/{prefix}_regex_patterns.json",
        "validation_report": f"results/{prefix}_validation_report.json",
        "final_report":      f"results/{prefix}_final_report.md",
        "debug_log":         f"logs/debug_{prefix}.log",
    }
