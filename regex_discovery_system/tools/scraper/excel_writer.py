"""Write scraped values to an Excel workbook for downstream agents."""
from __future__ import annotations

import logging
from pathlib import Path

import openpyxl

logger = logging.getLogger(__name__)


def write_to_excel(
    values: list[tuple[str, str]],
    condition_slug: str,
    output_dir: str = "data",
) -> str:
    """Persist scraped values as an xlsx file.

    Args:
        values:         List of (value, source_url) tuples.
        condition_slug: Slug used for the filename (e.g. ``california_zip_code``).
        output_dir:     Directory to write into (created if missing).

    Returns:
        Absolute path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{condition_slug}_scraped.xlsx"

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Scraped Data"
    ws.append(["value", "source_url"])
    for val, url in values:
        ws.append([val, url])
    wb.save(path)

    logger.info("[excel] wrote %d rows → %s", len(values), path)
    return str(path.resolve())
