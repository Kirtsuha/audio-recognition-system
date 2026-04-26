from pipeline.manifest import (
    iter_jsonl,
    iter_prepared_manifest,
    load_prepared_manifest,
    load_prepared_manifest_map,
    write_prepared_manifest,
)


def test_iter_jsonl_missing_returns_empty(tmp_path):
    rows = list(iter_jsonl(tmp_path / "missing.jsonl"))

    assert rows == []


def test_write_and_load_prepared_manifest(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rows = [
        {"track_id": 2, "split": "val", "prepared_path": "b.npy"},
        {"track_id": 1, "split": "train", "prepared_path": "a.npy"},
        {"track_id": 3, "split": "test", "prepared_path": "c.npy"},
    ]

    write_prepared_manifest(rows, path=path)

    loaded = list(iter_prepared_manifest(path))
    assert [x["track_id"] for x in loaded] == [1, 2, 3]

    limited = load_prepared_manifest(
        train_limit=1,
        val_limit=1,
        test_limit=1,
        path=path,
    )
    assert len(limited) == 3

    only = load_prepared_manifest(only_track_ids={2}, path=path)
    assert len(only) == 1
    assert only[0]["track_id"] == 2

    manifest_map = load_prepared_manifest_map(path=path)
    assert manifest_map[1]["prepared_path"] == "a.npy"