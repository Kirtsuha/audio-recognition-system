import json
from pathlib import Path
from typing import Iterable

from pipeline.config import PREPARED_MANIFEST_PATH
from pipeline.experiment_utils import apply_split_limits


def iter_jsonl(path: Path):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_prepared_manifest(path: Path = PREPARED_MANIFEST_PATH):
    yield from iter_jsonl(path)


def load_prepared_manifest(
    train_limit: int = 0,
    val_limit: int = 0,
    test_limit: int = 0,
    only_track_ids: set[int] | None = None,
    path: Path = PREPARED_MANIFEST_PATH,
) -> list[dict]:
    rows = list(iter_prepared_manifest(path))

    if only_track_ids is not None:
        rows = [row for row in rows if int(row["track_id"]) in only_track_ids]

    return apply_split_limits(
        rows,
        train_limit=train_limit,
        val_limit=val_limit,
        test_limit=test_limit,
    )


def load_all_prepared_manifest(
    only_track_ids: set[int] | None = None,
    path: Path = PREPARED_MANIFEST_PATH,
) -> list[dict]:
    rows = list(iter_prepared_manifest(path))

    if only_track_ids is not None:
        rows = [row for row in rows if int(row["track_id"]) in only_track_ids]

    rows.sort(key=lambda row: int(row["track_id"]))
    return rows


def load_prepared_manifest_map(path: Path = PREPARED_MANIFEST_PATH) -> dict[int, dict]:
    return {int(row["track_id"]): row for row in iter_prepared_manifest(path)}


def write_prepared_manifest(rows: Iterable[dict], path: Path = PREPARED_MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(rows, key=lambda row: int(row["track_id"]))

    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")