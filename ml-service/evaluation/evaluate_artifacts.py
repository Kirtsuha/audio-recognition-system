import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from app.model import AudioEncoder
from evaluation.evaluate_prepared import (
    PreparedAudioCache,
    evaluate_case,
    load_or_create_fixed_eval_items,
    resolve_eval_cases,
    resolve_primary_case,
)
from pipeline.config import FAISS_NPROBE
from pipeline.manifest import load_prepared_manifest

logger = logging.getLogger("ml-pipeline.evaluate-artifacts")


def _load_faiss_index(index_path: str | Path, faiss_meta_path: str | Path | None = None) -> faiss.Index:
    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))

    if hasattr(index, "nprobe"):
        nprobe = FAISS_NPROBE

        if faiss_meta_path is not None:
            meta_path = Path(faiss_meta_path)

            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)

                    if meta.get("nprobe") is not None:
                        nprobe = int(meta["nprobe"])

                except Exception:
                    logger.exception("Failed to read FAISS meta, fallback nprobe=%s", FAISS_NPROBE)

        index.nprobe = nprobe
        logger.info("Loaded IVF FAISS index nprobe=%s path=%s", nprobe, index_path)
    else:
        logger.info("Loaded FAISS index path=%s type=%s", index_path, type(index).__name__)

    return index


def _load_model(model_path: str | Path, device: torch.device) -> AudioEncoder:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = AudioEncoder().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model


def evaluate_artifacts(
    *,
    model_path: str | Path,
    index_path: str | Path,
    song_ids_path: str | Path,
    output_metrics_path: str | Path,
    faiss_meta_path: str | Path | None = None,
    output_results_path: str | Path | None = None,
    train_limit: int = 0,
    val_limit: int = 0,
    test_limit: int = 0,
    eval_query_limit: int = 500,
    use_full_query_audio: bool = False,
    fixed_eval_set_name: str = "prod_val500_v1",
    eval_noise_mode: str = "noisy",
    aggregation_strategies: list[str] | None = None,
) -> dict:


    started = time.time()

    output_metrics_path = Path(output_metrics_path)

    if output_results_path is None:
        output_results_path = output_metrics_path.parent / f"eval_results_{output_metrics_path.stem}.jsonl"
    else:
        output_results_path = Path(output_results_path)

    if output_results_path.exists():
        output_results_path.unlink()

    song_ids_path = Path(song_ids_path)

    if not song_ids_path.exists():
        raise FileNotFoundError(f"song_ids.npy not found: {song_ids_path}")

    song_ids = np.load(song_ids_path)

    index = _load_faiss_index(
        index_path=index_path,
        faiss_meta_path=faiss_meta_path,
    )

    if index.ntotal != len(song_ids):
        raise RuntimeError(
            f"Index/song_ids mismatch: index.ntotal={index.ntotal}, song_ids={len(song_ids)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = _load_model(model_path=model_path, device=device)

    index_items = load_prepared_manifest(
        train_limit=train_limit or 0,
        val_limit=val_limit or 0,
        test_limit=test_limit or 0,
    )

    if not index_items:
        raise ValueError("No prepared items available for artifact evaluation")

    val_items = [row for row in index_items if row["split"] == "val"]

    if val_limit and val_limit > 0:
        val_items = val_items[:val_limit]

    if len(val_items) < 5:
        raise ValueError(f"Need at least 5 validation queries, got {len(val_items)}")

    val_items = load_or_create_fixed_eval_items(
        val_items=val_items,
        eval_query_limit=eval_query_limit,
        fixed_eval_set_name=fixed_eval_set_name,
    )

    eval_cases = resolve_eval_cases(eval_noise_mode)
    primary_case_name = resolve_primary_case(eval_noise_mode, eval_cases)

    if aggregation_strategies is None or not aggregation_strategies:
        aggregation_strategies = ["max"]

    logger.info(
        (
            "Evaluate existing artifacts device=%s gpu=%s model=%s index=%s "
            "song_ids=%s fixed_eval_set=%s noise_mode=%s cases=%s primary_case=%s "
            "queries=%s strategies=%s"
        ),
        device,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        model_path,
        index_path,
        song_ids_path,
        fixed_eval_set_name,
        eval_noise_mode,
        eval_cases,
        primary_case_name,
        len(val_items),
        aggregation_strategies,
    )

    cache = PreparedAudioCache()
    cases: dict[str, dict[str, dict]] = {}

    for strategy in aggregation_strategies:
        strategy_cases: dict[str, dict] = {}

        for case_name in eval_cases:
            strategy_cases[case_name] = evaluate_case(
                model=model,
                index=index,
                song_ids=song_ids,
                val_items=val_items,
                cache=cache,
                device=device,
                use_full_query_audio=use_full_query_audio,
                case_name=case_name,
                aggregation_strategy=strategy,
                results_jsonl_path=output_results_path,
            )

        cases[strategy] = strategy_cases

    best_strategy = None
    best_case_metrics = None
    best_score = float("-inf")

    for strategy, strategy_cases in cases.items():
        case_metrics = strategy_cases.get(primary_case_name)

        if case_metrics is None:
            continue

        score = float(case_metrics.get("recall_at_1", 0.0))

        if score > best_score:
            best_score = score
            best_strategy = strategy
            best_case_metrics = case_metrics

    if best_strategy is None or best_case_metrics is None:
        best_strategy = aggregation_strategies[0]
        primary_case_name = next(iter(cases[best_strategy].keys()))
        best_case_metrics = cases[best_strategy][primary_case_name]

    metrics = {
        **best_case_metrics,
        "cases": cases,
        "primary_strategy": best_strategy,
        "primary_case": primary_case_name,
        "aggregation_strategies": aggregation_strategies,
        "eval_results_path": str(output_results_path),
        "model_path": str(model_path),
        "index_path": str(index_path),
        "song_ids_path": str(song_ids_path),
        "faiss_meta_path": str(faiss_meta_path) if faiss_meta_path else None,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "test_limit": test_limit,
        "eval_query_limit": eval_query_limit,
        "fixed_eval_set_name": fixed_eval_set_name,
        "eval_noise_mode": eval_noise_mode,
        "eval_cases": eval_cases,
        "index_vectors": int(index.ntotal),
        "use_full_query_audio": use_full_query_audio,
        "elapsed_sec": time.time() - started,
        "used_existing_index": True,
    }

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with output_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Artifact evaluation finished metrics=%s output=%s", metrics, output_metrics_path)

    return metrics