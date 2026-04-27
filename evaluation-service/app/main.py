from fastapi import FastAPI

from app.config import (
    COMBINED_RESULTS_PATH,
    EVAL_QUERIES_PATH,
    FINGERPRINT_RESULTS_PATH,
    METRICS_PATH,
    ML_RESULTS_PATH,
    PREPARED_MANIFEST_PATH,
)
from app.dataset import generate_eval_dataset
from app.metrics import build_all_metrics
from app.runners import (
    run_combined_evaluation,
    run_fingerprint_evaluation,
    run_ml_evaluation,
)
from app.schemas import EvalRunRequest, GenerateEvalDatasetRequest, RunAllRequest
from app.utils import read_jsonl

app = FastAPI(title="Music Recognition Evaluation Service")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "prepared_manifest_exists": PREPARED_MANIFEST_PATH.exists(),
        "eval_queries_exists": EVAL_QUERIES_PATH.exists(),
    }


@app.get("/evaluation/status")
def evaluation_status():
    return {
        "prepared_manifest_path": str(PREPARED_MANIFEST_PATH),
        "prepared_manifest_exists": PREPARED_MANIFEST_PATH.exists(),
        "eval_queries_path": str(EVAL_QUERIES_PATH),
        "eval_queries_exists": EVAL_QUERIES_PATH.exists(),
        "eval_queries_count": len(read_jsonl(EVAL_QUERIES_PATH)),
        "fingerprint_results_path": str(FINGERPRINT_RESULTS_PATH),
        "fingerprint_results_exists": FINGERPRINT_RESULTS_PATH.exists(),
        "ml_results_path": str(ML_RESULTS_PATH),
        "ml_results_exists": ML_RESULTS_PATH.exists(),
        "combined_results_path": str(COMBINED_RESULTS_PATH),
        "combined_results_exists": COMBINED_RESULTS_PATH.exists(),
        "metrics_path": str(METRICS_PATH),
        "metrics_exists": METRICS_PATH.exists(),
    }


@app.post("/evaluation/dataset/generate")
def generate_dataset_endpoint(payload: GenerateEvalDatasetRequest):
    return generate_eval_dataset(
        per_track_clean_queries=payload.per_track_clean_queries,
        per_track_noisy_queries=payload.per_track_noisy_queries,
        durations_sec=payload.durations_sec,
        test_limit=payload.test_limit,
        negative_limit=payload.negative_limit,
    )


@app.post("/evaluation/fingerprint/run")
def run_fingerprint_endpoint(payload: EvalRunRequest):
    return run_fingerprint_evaluation(
        limit=payload.limit,
        timeout_sec=payload.timeout_sec,
    )


@app.post("/evaluation/ml/run")
def run_ml_endpoint(payload: EvalRunRequest):
    return run_ml_evaluation(
        limit=payload.limit,
        timeout_sec=payload.timeout_sec,
    )


@app.post("/evaluation/combined/run")
def run_combined_endpoint(payload: EvalRunRequest):
    return run_combined_evaluation(
        limit=payload.limit,
        timeout_sec=payload.timeout_sec,
    )


@app.post("/evaluation/metrics/build")
def build_metrics_endpoint():
    return build_all_metrics()


@app.post("/evaluation/run-all")
def run_all_endpoint(payload: RunAllRequest):
    result = {}

    if payload.generate_dataset:
        result["dataset"] = generate_eval_dataset(
            per_track_clean_queries=payload.per_track_clean_queries,
            per_track_noisy_queries=payload.per_track_noisy_queries,
            durations_sec=payload.durations_sec,
            test_limit=payload.test_limit,
            negative_limit=payload.negative_limit,
        )

    result["fingerprint"] = run_fingerprint_evaluation(
        limit=payload.eval_limit,
        timeout_sec=payload.timeout_sec,
    )

    result["ml"] = run_ml_evaluation(
        limit=payload.eval_limit,
        timeout_sec=payload.timeout_sec,
    )

    result["combined"] = run_combined_evaluation(
        limit=payload.eval_limit,
        timeout_sec=payload.timeout_sec,
    )

    result["metrics"] = build_all_metrics()

    return result