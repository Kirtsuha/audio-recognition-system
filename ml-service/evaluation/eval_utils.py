import json
import math
from pathlib import Path
from typing import Iterable


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    values = sorted(values)
    rank = (len(values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)

    if low == high:
        return float(values[low])

    frac = rank - low
    return float(values[low] * (1 - frac) + values[high] * frac)