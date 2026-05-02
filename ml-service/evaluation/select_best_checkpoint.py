import json
import logging
import re
import shutil
from pathlib import Path

from evaluation.evaluate_prepared import evaluate_prepared

logger = logging.getLogger("ml-pipeline.best-checkpoint")


def _epoch_num(path: Path) -> int:
    match = re.search(r"model_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def select_best_checkpoint(
    run_dir: str | Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    eval_query_limit: int,
    index_windows_override: int,
    use_full_query_audio: bool,
    fixed_eval_set_name: str,
    eval_noise_mode: str = "noisy",
    metric_name: str = "recall_at_1",
) -> dict:
    run_dir = Path(run_dir)
    checkpoints = sorted(run_dir.glob("model_epoch_*.pt"), key=_epoch_num)

    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found in {run_dir}")

    results = []
    best = None

    for ckpt in checkpoints:
        metrics_path = run_dir / f"metrics_{ckpt.stem}.json"

        metrics = evaluate_prepared(
            model_path=ckpt,
            output_metrics_path=metrics_path,
            train_limit=train_limit,
            val_limit=val_limit,
            test_limit=test_limit,
            eval_query_limit=eval_query_limit,
            index_windows_override=index_windows_override,
            use_full_query_audio=use_full_query_audio,
            fixed_eval_set_name=fixed_eval_set_name,
            eval_noise_mode=eval_noise_mode,
        )

        score = float(metrics.get(metric_name, 0.0))

        item = {
            "checkpoint": str(ckpt),
            "epoch": _epoch_num(ckpt),
            "metric_name": metric_name,
            "score": score,
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        }

        results.append(item)

        if best is None or score > best["score"]:
            best = item

        logger.info(
            "Checkpoint evaluated epoch=%s score=%.4f metric=%s path=%s",
            item["epoch"],
            score,
            metric_name,
            ckpt,
        )

    assert best is not None

    best_model_path = run_dir / "model_best.pt"
    shutil.copy2(best["checkpoint"], best_model_path)

    summary = {
        "best": {
            **best,
            "best_model_path": str(best_model_path),
        },
        "results": results,
    }

    summary_path = run_dir / "best_checkpoint_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Best checkpoint selected summary=%s", summary["best"])

    return summary