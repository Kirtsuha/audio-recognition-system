import json

import numpy as np

from audio.preprocess import normalize
from fingerprint_experiment.metrics import build_metrics
from fingerprint_experiment.utils import percentile, write_json, write_jsonl


def test_normalize_empty_and_peak():
    assert normalize(np.array([], dtype=np.float32)).size == 0

    result = normalize(np.array([0.0, 2.0, -1.0], dtype=np.float32))

    assert result.dtype == np.float32
    assert result.tolist() == [0.0, 1.0, -0.5]


def test_percentile_interpolates_and_ignores_none():
    assert percentile([], 0.5) is None
    assert percentile([None, 10], 0.95) == 10.0
    assert percentile([0, 10], 0.5) == 5.0


def test_write_json_and_jsonl(tmp_path):
    json_path = tmp_path / "nested" / "payload.json"
    jsonl_path = tmp_path / "nested" / "rows.jsonl"

    write_json(json_path, {"ok": True})
    write_jsonl(jsonl_path, [{"a": 1}, {"b": "test"}])

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    assert jsonl_path.read_text(encoding="utf-8").splitlines() == ['{"a": 1}', '{"b": "test"}']


def test_build_metrics_groups_rows():
    rows = [
        {
            "is_positive": True,
            "correct_top1": True,
            "correct_top5": True,
            "matched": True,
            "latency_ms": 10,
            "confidence": 0.9,
            "query_hashes": 100,
            "aligned_matches": 15,
            "bucket": "clean",
            "corruption": "none",
            "duration_sec": 5.0,
        },
        {
            "is_positive": False,
            "correct_top1": False,
            "correct_top5": False,
            "matched": True,
            "latency_ms": 30,
            "confidence": 0.2,
            "query_hashes": 50,
            "aligned_matches": 3,
            "bucket": "negative",
            "corruption": "noise",
            "duration_sec": 5.0,
        },
    ]

    summary = build_metrics(rows)

    assert summary["overall"]["queries"] == 2
    assert summary["overall"]["accuracy_at_1"] == 1.0
    assert summary["overall"]["false_positive_rate"] == 1.0
    assert summary["by_bucket"]["clean"]["positives"] == 1
