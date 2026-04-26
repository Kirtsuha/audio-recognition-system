from evaluation.eval_utils import percentile, read_jsonl, write_jsonl


def test_write_and_read_jsonl(tmp_path):
    path = tmp_path / "data" / "rows.jsonl"
    rows = [{"a": 1}, {"b": "тест"}]

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows


def test_read_jsonl_missing_file_returns_empty(tmp_path):
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_percentile_empty():
    assert percentile([], 0.5) is None


def test_percentile_single_value():
    assert percentile([10], 0.95) == 10.0


def test_percentile_interpolates():
    assert percentile([0, 10], 0.5) == 5.0
    assert percentile([0, 10, 20], 0.5) == 10.0