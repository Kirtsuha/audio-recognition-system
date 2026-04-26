import logging
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException

from app.logging_utils import configure_logging, log_stage
from pipeline_runner.schemas import AsyncJobResponse, PipelineRequest, ExperimentRunRequest
from pipeline.build_embeddings_from_prepared import build_embeddings_from_prepared
from pipeline.build_index import build_faiss_index
from pipeline.incremental_sync import build_incremental_run
from pipeline.config import RUNS_DIR
from evaluation.evaluate_prepared import evaluate_prepared
from pipeline.job_status import get_status, reset_progress, update_status
from pipeline.prepare_data import prepare_data
from pipeline.promote import promote_run_to_active
from pipeline.train_prepared import train_prepared

app = FastAPI(title="ML Pipeline Runner")
configure_logging()
logger = logging.getLogger("ml-pipeline-runner")

pipeline_in_progress = False


def make_run_dir(prefix: str) -> Path:
    run_id = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_full_pipeline(bucket: str, prefix: str) -> None:
    global pipeline_in_progress
    if pipeline_in_progress:
        return

    pipeline_in_progress = True
    run_dir = make_run_dir("full")

    update_status(
        current_job="full-pipeline",
        phase="prepare-data",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={"run_dir": str(run_dir)},
    )
    reset_progress()

    try:
        with log_stage(logger, "prepare-data", bucket=bucket, prefix=prefix):
            prepare_summary = prepare_data(bucket=bucket, prefix=prefix, append=False)

        update_status(phase="train")
        model_path = run_dir / "model.pt"
        with log_stage(logger, "train-prepared", model_path=str(model_path)):
            train_prepared(output_model_path=model_path)

        update_status(phase="evaluate")
        metrics_path = run_dir / "metrics.json"
        with log_stage(logger, "evaluate-prepared", metrics_path=str(metrics_path)):
            evaluate_prepared(model_path=model_path, output_metrics_path=metrics_path)

        update_status(phase="build-embeddings")
        embeddings_path = run_dir / "embeddings.npy"
        song_ids_path = run_dir / "song_ids.npy"
        manifest_path = run_dir / "song_manifest.json"
        with log_stage(logger, "build-embeddings-from-prepared"):
            build_embeddings_from_prepared(
                model_path=model_path,
                embeddings_out=embeddings_path,
                song_ids_out=song_ids_path,
                manifest_out=manifest_path,
            )

        update_status(phase="build-index")
        index_path = run_dir / "faiss.index"
        with log_stage(logger, "build-faiss-index"):
            build_faiss_index(
                embeddings_path=embeddings_path,
                song_ids_path=song_ids_path,
                index_out_path=index_path,
            )

        update_status(phase="promote")
        with log_stage(logger, "promote-active", run_dir=str(run_dir)):
            promote_run_to_active(run_dir)

        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={"run_dir": str(run_dir), "prepare_summary": prepare_summary},
        )
    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={"run_dir": str(run_dir)},
        )
        logger.exception("Full pipeline failed")
    finally:
        pipeline_in_progress = False


def run_experiment_pipeline(payload: ExperimentRunRequest) -> None:
    global pipeline_in_progress
    if pipeline_in_progress:
        return

    pipeline_in_progress = True
    run_dir = make_run_dir(f"exp_{payload.experiment_name}")

    update_status(
        current_job="experiment-pipeline",
        phase="prepare-data",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={
            "run_dir": str(run_dir),
            "experiment_name": payload.experiment_name,
            "train_limit": payload.train_limit,
            "val_limit": payload.val_limit,
            "test_limit": payload.test_limit,
            "epochs_override": payload.epochs_override,
            "eval_query_limit": payload.eval_query_limit,
            "index_windows_override": payload.index_windows_override,
            "use_full_query_audio": payload.use_full_query_audio,
        },
    )
    reset_progress()

    try:
        with log_stage(logger, "prepare-data", bucket=payload.bucket, prefix=payload.prefix):
            if payload.skip_prepare:
                prepare_summary = {"skipped": True}
                logger.info("Prepare-data skipped by request")
            else:
                with log_stage(logger, "prepare-data", bucket=payload.bucket, prefix=payload.prefix):
                    prepare_summary = prepare_data(
                        bucket=payload.bucket,
                        prefix=payload.prefix,
                        append=False,
                        prepare_limit=payload.prepare_limit,
                    )

        update_status(phase="train")
        model_path = run_dir / "model.pt"
        with log_stage(logger, "train-prepared", model_path=str(model_path)):
            train_prepared(
                output_model_path=model_path,
                train_limit=payload.train_limit,
                epochs_override=payload.epochs_override,
            )

        update_status(phase="evaluate")
        metrics_path = run_dir / "metrics.json"
        with log_stage(logger, "evaluate-prepared", metrics_path=str(metrics_path)):
            evaluate_prepared(
                model_path=model_path,
                output_metrics_path=metrics_path,
                train_limit=payload.train_limit,
                val_limit=payload.val_limit,
                test_limit=payload.test_limit,
                eval_query_limit=payload.eval_query_limit,
                index_windows_override=payload.index_windows_override,
                use_full_query_audio=payload.use_full_query_audio
            )

        update_status(phase="build-embeddings")
        embeddings_path = run_dir / "embeddings.npy"
        song_ids_path = run_dir / "song_ids.npy"
        manifest_path = run_dir / "song_manifest.json"
        with log_stage(logger, "build-embeddings-from-prepared"):
            build_embeddings_from_prepared(
                model_path=model_path,
                embeddings_out=embeddings_path,
                song_ids_out=song_ids_path,
                manifest_out=manifest_path,
                train_limit=payload.train_limit,
                val_limit=payload.val_limit,
                test_limit=payload.test_limit,
                index_windows_override=payload.index_windows_override,
            )

        update_status(phase="build-index")
        index_path = run_dir / "faiss.index"
        with log_stage(logger, "build-faiss-index"):
            build_faiss_index(
                embeddings_path=embeddings_path,
                song_ids_path=song_ids_path,
                index_out_path=index_path,
            )

        # Для experiment-run не промоутим в active автоматически
        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={
                "run_dir": str(run_dir),
                "prepare_summary": prepare_summary,
                "experiment_name": payload.experiment_name,
                "train_limit": payload.train_limit,
                "val_limit": payload.val_limit,
                "test_limit": payload.test_limit,
                "epochs_override": payload.epochs_override,
                "eval_query_limit": payload.eval_query_limit,
                "index_windows_override": payload.index_windows_override,
                "promoted": False,
                "prepare_limit": payload.prepare_limit,
                "skip_prepare": payload.skip_prepare,
            },
        )
    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={"run_dir": str(run_dir), "experiment_name": payload.experiment_name},
        )
        logger.exception("Experiment pipeline failed")
    finally:
        pipeline_in_progress = False


def run_incremental_pipeline(bucket: str, prefix: str) -> None:
    global pipeline_in_progress
    if pipeline_in_progress:
        return

    pipeline_in_progress = True
    run_dir = make_run_dir("incremental")

    update_status(
        current_job="incremental-pipeline",
        phase="detect-new-tracks",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={"run_dir": str(run_dir)},
    )
    reset_progress()

    try:
        update_status(phase="incremental-sync")
        with log_stage(logger, "incremental-sync", bucket=bucket, prefix=prefix):
            summary = build_incremental_run(bucket=bucket, prefix=prefix, run_dir=run_dir)

        if summary.get("no_changes"):
            update_status(
                phase="done",
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                progress={"run_dir": str(run_dir), "summary": summary},
            )
            return

        update_status(phase="promote")
        with log_stage(logger, "promote-active", run_dir=str(run_dir)):
            promote_run_to_active(run_dir)

        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={"run_dir": str(run_dir), "summary": summary},
        )
    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={"run_dir": str(run_dir)},
        )
        logger.exception("Incremental pipeline failed")
    finally:
        pipeline_in_progress = False


@app.post("/pipeline/full-run", response_model=AsyncJobResponse)
async def full_run(payload: PipelineRequest, background_tasks: BackgroundTasks):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_full_pipeline, payload.bucket, payload.prefix)
    return AsyncJobResponse(accepted=True, status="scheduled", bucket=payload.bucket, prefix=payload.prefix)


@app.post("/pipeline/experiment-run", response_model=AsyncJobResponse)
async def experiment_run(payload: ExperimentRunRequest, background_tasks: BackgroundTasks):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_experiment_pipeline, payload)
    return AsyncJobResponse(accepted=True, status="scheduled", bucket=payload.bucket, prefix=payload.prefix)


@app.post("/pipeline/incremental-sync", response_model=AsyncJobResponse)
async def incremental_sync(payload: PipelineRequest, background_tasks: BackgroundTasks):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_incremental_pipeline, payload.bucket, payload.prefix)
    return AsyncJobResponse(accepted=True, status="scheduled", bucket=payload.bucket, prefix=payload.prefix)


@app.get("/pipeline/status")
async def pipeline_status():
    return {
        "pipeline_in_progress": pipeline_in_progress,
        "job_status": get_status(),
    }


@app.get("/health")
async def health():
    return {
        "status": "busy" if pipeline_in_progress else "ok",
        "pipeline_in_progress": pipeline_in_progress,
    }