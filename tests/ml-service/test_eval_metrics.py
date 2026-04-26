import json

import evaluation.eval_metrics as metrics
from evaluation.eval_utils import write_jsonl


def test_build_eval_metrics(tmp_path, monkeypatch):
    results_path = tmp_path / "eval_results.jsonl"
    metrics_path = tmp_path / "eval_metrics.json"

    rows = [
        {
            "is_positive": True,
            "correct_top1": True,
            "correct_top5": True,
            "matched": True,
            "latency_ms": 100,
            "confidence": 0.9,
            "margin": 0.5,
            "bucket": "clean",
            "corruption": "clean",
            "duration_sec": 5.0,
        },
        {
            "is_positive": True,
            "correct_top1": False,
            "correct_top5": True,
            "matched": False,
            "latency_ms": 200,
            "confidence": 0.7,
            "margin": 0.2,
            "bucket": "dirty",
            "corruption": "noise",
            "duration_sec": 10.0,
        },
        {
            "is_positive": False,
            "correct_top1": False,
            "correct_top5": False,
            "matched": True,
            "latency_ms": 300,
            "confidence": None,
            "margin": None,
            "bucket": "negative",
            "corruption": "white_noise",
            "duration_sec": 5.0,
        },
    ]

    write_jsonl(results_path, rows)

    monkeypatch.setattr(metrics, "EVAL_RESULTS_PATH", results_path)
    monkeypatch.setattr(metrics, "EVAL_METRICS_PATH", metrics_path)

    summary = metrics.build_eval_metrics()

    assert summary["overall"]["queries"] == 3
    assert summary["overall"]["positives"] == 2
    assert summary["overall"]["negatives"] == 1
    assert summary["overall"]["accuracy_at_1"] == 0.5
    assert summary["overall"]["recall_at_5"] == 1.0
    assert summary["overall"]["false_positive_rate"] == 1.0
    assert metrics_path.exists()

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["overall"]["queries"] == 3


def test_build_eval_metrics_empty_raises(tmp_path, monkeypatch):
    results_path = tmp_path / "empty.jsonl"
    results_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(metrics, "EVAL_RESULTS_PATH", results_path)

    try:
        metrics.build_eval_metrics()
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")