import json
import math
from pathlib import Path
from typing import Iterable


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_parent(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    ensure_parent(path)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[float], p: float) -> float | None:
    values = [float(x) for x in values if x is not None]

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


def safe_int(value) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def safe_float(value) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None