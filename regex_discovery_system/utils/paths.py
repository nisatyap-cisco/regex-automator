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
        "regex_patterns":    f"results/{prefix}_regex_patterns.json",
        "debug_log":         f"logs/debug_{prefix}.log",
    }
