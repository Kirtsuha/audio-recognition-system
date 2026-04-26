import random
from typing import Iterable

from pipeline.config import EXPERIMENT_RANDOM_SEED


def stable_limit_by_track_id(items: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or limit >= len(items):
        return list(items)

    items_sorted = sorted(items, key=lambda x: int(x["track_id"]))
    return items_sorted[:limit]


def apply_split_limits(
    items: list[dict],
    train_limit: int = 0,
    val_limit: int = 0,
    test_limit: int = 0,
) -> list[dict]:
    train_items = [x for x in items if x["split"] == "train"]
    val_items = [x for x in items if x["split"] == "val"]
    test_items = [x for x in items if x["split"] == "test"]

    train_items = stable_limit_by_track_id(train_items, train_limit)
    val_items = stable_limit_by_track_id(val_items, val_limit)
    test_items = stable_limit_by_track_id(test_items, test_limit)

    result = train_items + val_items + test_items
    result.sort(key=lambda x: (x["split"], int(x["track_id"])))
    return result


def maybe_limit_rows(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    return rows[:limit]


def set_experiment_seed(seed: int = EXPERIMENT_RANDOM_SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass