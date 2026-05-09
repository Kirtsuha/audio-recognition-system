import json

import evaluation.threshold_tuning as tuning
from evaluation.eval_utils import write_jsonl


def test_tune_thresholds(tmp_path, monkeypatch):
    results_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "threshold_tuning.json"

    rows = [
        {
            "is_positive": True,
            "confidence": 0.9,
            "margin": 0.4,
            "support": 3,
            "correct_top1": True,
            "correct_top5": True,
        },
        {
            "is_positive": True,
            "confidence": 0.4,
            "margin": 0.1,
            "support": 1,
            "correct_top1": False,
            "correct_top5": True,
        },
        {
            "is_positive": False,
            "confidence": 0.95,
            "margin": 0.5,
            "support": 4,
            "correct_top1": False,
            "correct_top5": False,
        },
    ]

    write_jsonl(results_path, rows)
    monkeypatch.setattr(tuning, "THRESHOLD_TUNING_PATH", output_path)

    summary = tuning.tune_thresholds(
        confidence_values=[0.5, 0.8],
        margin_values=[0.2],
        support_values=[2],
        results_path=str(results_path),
    )

    assert summary["best"] is not None
    assert summary["total_candidates"] == 2
    assert output_path.exists()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["total_candidates"] == 2


def test_tune_thresholds_empty_raises(tmp_path):
    results_path = tmp_path / "empty.jsonl"
    results_path.write_text("", encoding="utf-8")

    try:
        tuning.tune_thresholds([0.5], [0.1], [1], results_path=str(results_path))
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")