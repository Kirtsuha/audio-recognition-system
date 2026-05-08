import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.config import (
    COMBINED_RESULTS_PATH,
    EVAL_AUDIO_SOURCE,
    EVAL_QUERIES_PATH,
    FINGERPRINT_RESULTS_PATH,
    METRICS_PATH,
    ORCHESTRATION_SERVICE_URL,
    PREPARED_MANIFEST_PATH,
    RECOGNITION_RESULTS_PATH,
    S3_BUCKET,
    S3_ENDPOINT,
    SR,
)
from app.dataset import generate_eval_dataset
from app.logging_utils import configure_logging, log_stage
from app.metrics import build_all_metrics
from app.runners import (
    run_combined_evaluation,
    run_fingerprint_evaluation,
    run_recognition_evaluation,
)
from app.schemas import EvalRunRequest, GenerateEvalDatasetRequest, RunAllRequest
from app.utils import read_jsonl

configure_logging()
logger = logging.getLogger("evaluation-service")

app = FastAPI(title="Music Recognition Evaluation Service")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "prepared_manifest_exists": PREPARED_MANIFEST_PATH.exists(),
        "eval_queries_exists": EVAL_QUERIES_PATH.exists(),
        "audio_source": EVAL_AUDIO_SOURCE,
        "sr": SR,
        "s3_endpoint": S3_ENDPOINT,
        "s3_bucket": S3_BUCKET,
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
        "recognition_results_path": str(RECOGNITION_RESULTS_PATH),
        "recognition_results_exists": RECOGNITION_RESULTS_PATH.exists(),
        "combined_results_path": str(COMBINED_RESULTS_PATH),
        "combined_results_exists": COMBINED_RESULTS_PATH.exists(),
        "metrics_path": str(METRICS_PATH),
        "metrics_exists": METRICS_PATH.exists(),
        "audio_source": EVAL_AUDIO_SOURCE,
        "sr": SR,
        "s3_endpoint": S3_ENDPOINT,
        "s3_bucket": S3_BUCKET,
        "orchestration_url": ORCHESTRATION_SERVICE_URL,
    }


@app.post("/evaluation/dataset/generate")
def generate_dataset_endpoint(payload: GenerateEvalDatasetRequest):
    logger.info("Request: generate dataset payload=%s", payload.model_dump())

    return generate_eval_dataset(
        per_track_clean_queries=payload.per_track_clean_queries,
        per_track_noisy_queries=payload.per_track_noisy_queries,
        durations_sec=payload.durations_sec,
        test_limit=payload.test_limit,
        negative_limit=payload.negative_limit,
        corruptions=payload.corruptions,
        seed=payload.seed,
    )


@app.post("/evaluation/fingerprint/run")
def run_fingerprint_endpoint(payload: EvalRunRequest):
    logger.info("Request: run fingerprint evaluation payload=%s", payload.model_dump())

    return run_fingerprint_evaluation(
        limit=payload.limit,
        timeout_sec=payload.timeout_sec,
        top_k=payload.top_k,
    )


@app.post("/evaluation/recognition/run")
def run_recognition_endpoint(payload: EvalRunRequest):
    logger.info("Request: run recognition evaluation payload=%s", payload.model_dump())

    return run_recognition_evaluation(
        limit=payload.limit,
        timeout_sec=payload.timeout_sec,
    )


@app.post("/evaluation/combined/run")
def run_combined_endpoint(payload: EvalRunRequest):
    logger.info("Request: run combined evaluation payload=%s", payload.model_dump())

    return run_combined_evaluation(
        limit=payload.limit,
        timeout_sec=payload.timeout_sec,
        top_k=payload.top_k,
    )


@app.post("/evaluation/metrics/build")
def build_metrics_endpoint():
    logger.info("Request: build metrics")
    return build_all_metrics()


@app.post("/evaluation/run-all")
def run_all_endpoint(payload: RunAllRequest):
    logger.info("Request: run all payload=%s", payload.model_dump())

    with log_stage(logger, "run-all"):
        result = {}

        if payload.generate_dataset:
            result["dataset"] = generate_eval_dataset(
                per_track_clean_queries=payload.per_track_clean_queries,
                per_track_noisy_queries=payload.per_track_noisy_queries,
                durations_sec=payload.durations_sec,
                test_limit=payload.test_limit,
                negative_limit=payload.negative_limit,
                corruptions=payload.corruptions,
                seed=payload.seed,
            )

        result["fingerprint"] = run_fingerprint_evaluation(
            limit=payload.eval_limit,
            timeout_sec=payload.timeout_sec,
            top_k=payload.top_k,
        )

        result["recognition"] = run_recognition_evaluation(
            limit=payload.eval_limit,
            timeout_sec=payload.timeout_sec,
        )

        result["combined"] = run_combined_evaluation(
            limit=payload.eval_limit,
            timeout_sec=payload.timeout_sec,
            top_k=payload.top_k,
        )

        result["metrics"] = build_all_metrics()

        logger.info("Run all finished")
        return result
