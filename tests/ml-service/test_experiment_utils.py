from pipeline.experiment_utils import (
    apply_split_limits,
    maybe_limit_rows,
    stable_limit_by_track_id,
)


def test_stable_limit_by_track_id():
    items = [{"track_id": 3}, {"track_id": 1}, {"track_id": 2}]

    result = stable_limit_by_track_id(items, 2)

    assert [x["track_id"] for x in result] == [1, 2]


def test_stable_limit_no_limit_returns_copy():
    items = [{"track_id": 1}, {"track_id": 2}]

    result = stable_limit_by_track_id(items, 0)

    assert result == items
    assert result is not items


def test_apply_split_limits():
    items = [
        {"track_id": 3, "split": "train"},
        {"track_id": 1, "split": "train"},
        {"track_id": 2, "split": "val"},
        {"track_id": 4, "split": "test"},
    ]

    result = apply_split_limits(items, train_limit=1, val_limit=1, test_limit=1)

    assert result == [
        {"track_id": 4, "split": "test"},
        {"track_id": 1, "split": "train"},
        {"track_id": 2, "split": "val"},
    ]


def test_maybe_limit_rows():
    rows = [{"x": 1}, {"x": 2}, {"x": 3}]

    assert maybe_limit_rows(rows, 2) == [{"x": 1}, {"x": 2}]
    assert maybe_limit_rows(rows, 0) == rows