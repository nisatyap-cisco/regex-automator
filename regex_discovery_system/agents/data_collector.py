from __future__ import annotations

import logging
import math
import random

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a data-generation assistant.
Generate {batch_size} UNIQUE, REALISTIC examples of: {condition}.
Rules:
- One value per line, no numbering, no bullet points, no extra text.
- Values must resemble real-world data actually in use.
- Do not repeat values from previous batches: {previous_values_sample}
Output ONLY the values."""

SEED = 42


def _write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def collect_data(condition: str, config: dict) -> dict:
    dataset_size = config.get("dataset_size", 1000)
    batch_size = config.get("batch_size", 200)
    train_ratio = config.get("train_ratio", 0.70)
    num_batches = math.ceil(dataset_size / batch_size)

    logger.info(
        "Agent 1: collecting %d examples in %d batches of %d",
        dataset_size,
        num_batches,
        batch_size,
    )

    all_values: list[str] = []

    for batch_idx in range(num_batches):
        sample_prev = ", ".join(all_values[-50:]) if all_values else "N/A (first batch)"
        prompt = PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            condition=condition,
            previous_values_sample=sample_prev,
        )

        try:
            response = invoke_claude(prompt, config)
        except Exception as exc:
            logger.warning("Batch %d/%d failed after retries: %s", batch_idx + 1, num_batches, exc)
            continue

        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        all_values.extend(lines)
        logger.info("Batch %d/%d: got %d values (total raw: %d)", batch_idx + 1, num_batches, len(lines), len(all_values))

    seen: set[str] = set()
    unique: list[str] = []
    for v in all_values:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    if len(unique) < dataset_size:
        logger.warning("Only collected %d unique values (target was %d)", len(unique), dataset_size)

    rng = random.Random(SEED)
    rng.shuffle(unique)

    split_idx = int(len(unique) * train_ratio)
    train = unique[:split_idx]
    test = unique[split_idx:]

    _write_lines("data/raw_examples.txt", unique)
    _write_lines("data/train.txt", train)
    _write_lines("data/test.txt", test)

    logger.info(
        "Agent 1 done: %d unique → %d train / %d test",
        len(unique),
        len(train),
        len(test),
    )

    return {"total": len(unique), "train": len(train), "test": len(test)}
