import logging
import time
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query

from app.logging_utils import configure_logging, log_stage
from evaluation.evaluate_artifacts import evaluate_artifacts
from evaluation.reranker_evaluation import evaluate_reranker
from evaluation.threshold_tuning import tune_thresholds
from pipeline.build_reranker_referece_embeddings import build_reranker_reference_embeddings
from pipeline.cleanup import cleanup_runs
from pipeline.retrain_status import get_retrain_status
from pipeline.run_metadata import write_run_metadata
from pipeline_runner.schemas import (
    AsyncJobResponse,
    ExperimentRunRequest,
    ThresholdTuneExperimentRequest,
    FullRunRequest, EvaluateExistingModelRequest, IncrementalSyncRequest, PromoteRunRequest,
    CleanupRequest, BuildEmbeddingsRequest, BuildFaissOnlyRequest, EvaluateArtifactsRequest, EvaluateRerankerRequest,
    BuildRerankerReferenceEmbeddingsRequest
)
from pipeline.build_embeddings_from_prepared import build_embeddings_from_prepared
from pipeline.build_index import build_faiss_index
from pipeline.incremental_sync import build_incremental_run
from pipeline.config import RUNS_DIR
from evaluation.evaluate_prepared import evaluate_prepared
from evaluation.select_best_checkpoint import select_best_checkpoint
from pipeline.job_status import get_status, reset_progress, update_status
from pipeline.prepare_data import prepare_data
from pipeline.promote import promote_run_to_active, validate_run_artifacts
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


def run_full_pipeline(payload: FullRunRequest) -> None:
    global pipeline_in_progress
    if pipeline_in_progress:
        return

    pipeline_in_progress = True
    run_dir = make_run_dir("full")

    update_status(
        current_job="full-build",
        phase="prepare-data",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={
            "run_dir": str(run_dir),
            "train_limit": payload.train_limit,
            "val_limit": payload.val_limit,
            "test_limit": payload.test_limit,
            "prepare_limit": payload.prepare_limit,
            "index_all_prepared": payload.index_all_prepared,
            "index_windows_override": payload.index_windows_override,
            "aggregation_strategies": payload.aggregation_strategies,
            "promote": payload.promote,
        },
    )
    reset_progress()

    best_summary = None
    metrics = None
    embed_summary = None
    index_summary = None
    promoted = False
    warnings = []

    try:
        with log_stage(logger, "prepare-data", bucket=payload.bucket, prefix=payload.prefix):
            if payload.skip_prepare:
                prepare_summary = {"skipped": True}
                logger.info("Prepare-data skipped by request")
            else:
                prepare_summary = prepare_data(
                    bucket=payload.bucket,
                    prefix=payload.prefix,
                    append=True,
                    prepare_limit=payload.prepare_limit,
                    skip_existing=True,
                )

        update_status(phase="train")
        raw_model_path = run_dir / "model_last.pt"

        with log_stage(logger, "train-prepared", model_path=str(raw_model_path)):
            train_prepared(
                output_model_path=raw_model_path,
                train_limit=payload.train_limit,
                epochs_override=payload.epochs_override,
            )

        model_path = raw_model_path

        if payload.select_best_checkpoint:
            update_status(phase="select-best-checkpoint")

            with log_stage(logger, "select-best-checkpoint", run_dir=str(run_dir)):
                best_summary = select_best_checkpoint(
                    run_dir=run_dir,
                    train_limit=payload.train_limit,
                    val_limit=payload.val_limit,
                    test_limit=payload.test_limit,
                    eval_query_limit=payload.eval_query_limit,
                    index_windows_override=payload.index_windows_override,
                    use_full_query_audio=payload.use_full_query_audio,
                    fixed_eval_set_name=payload.fixed_eval_set_name,
                    eval_noise_mode=payload.best_checkpoint_eval_noise_mode,
                    metric_name=payload.best_checkpoint_metric,
                )

            model_path = run_dir / "model_best.pt"

        final_model_path = run_dir / "model.pt"
        if model_path.resolve() != final_model_path.resolve():
            shutil.copy2(model_path, final_model_path)
        model_path = final_model_path

        update_status(phase="evaluate")
        metrics_path = run_dir / "metrics.json"
        eval_results_path = run_dir / "eval_results.jsonl"

        with log_stage(logger, "evaluate-prepared", metrics_path=str(metrics_path)):
            metrics = evaluate_prepared(
                model_path=model_path,
                output_metrics_path=metrics_path,
                train_limit=payload.train_limit,
                val_limit=payload.val_limit,
                test_limit=payload.test_limit,
                eval_query_limit=payload.eval_query_limit,
                index_windows_override=payload.index_windows_override,
                use_full_query_audio=payload.use_full_query_audio,
                fixed_eval_set_name=payload.fixed_eval_set_name,
                eval_noise_mode=payload.eval_noise_mode,
                results_jsonl_path=eval_results_path,
                aggregation_strategies=payload.aggregation_strategies,
            )

        update_status(phase="build-embeddings")
        embeddings_path = run_dir / "embeddings.npy"
        song_ids_path = run_dir / "song_ids.npy"
        manifest_path = run_dir / "song_manifest.json"

        with log_stage(logger, "build-embeddings-from-prepared"):
            embed_summary = build_embeddings_from_prepared(
                model_path=model_path,
                embeddings_out=embeddings_path,
                song_ids_out=song_ids_path,
                manifest_out=manifest_path,
                train_limit=payload.train_limit,
                val_limit=payload.val_limit,
                test_limit=payload.test_limit,
                index_windows_override=payload.index_windows_override,
                index_all_prepared=payload.index_all_prepared,
            )

        update_status(phase="build-index")
        index_path = run_dir / "faiss.index"
        index_meta_path = run_dir / "faiss_meta.json"

        with log_stage(logger, "build-faiss-index"):
            index_summary = build_faiss_index(
                embeddings_path=embeddings_path,
                song_ids_path=song_ids_path,
                index_out_path=index_path,
                meta_out_path=index_meta_path,
            )

        if not payload.promote:
            warnings.append("Full build finished but was not promoted. Use /pipeline/promote-run after reviewing metrics.")

        metadata = write_run_metadata(
            run_dir=run_dir,
            run_type="full-build",
            model_path=model_path,
            metrics=metrics,
            prepare_summary=prepare_summary,
            best_summary=best_summary,
            embed_summary=embed_summary,
            index_summary=index_summary,
            payload=payload.model_dump(),
            promoted=False,
            warnings=warnings,
        )

        if payload.promote:
            update_status(phase="promote")
            with log_stage(logger, "promote-active", run_dir=str(run_dir)):
                promote_run_to_active(run_dir)
            promoted = True

            metadata = write_run_metadata(
                run_dir=run_dir,
                run_type="full-build",
                model_path=model_path,
                metrics=metrics,
                prepare_summary=prepare_summary,
                best_summary=best_summary,
                embed_summary=embed_summary,
                index_summary=index_summary,
                payload=payload.model_dump(),
                promoted=True,
                warnings=warnings,
            )

        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={
                "run_dir": str(run_dir),
                "prepare_summary": prepare_summary,
                "best_summary": best_summary,
                "metrics": metrics,
                "embed_summary": embed_summary,
                "index_summary": index_summary,
                "metadata": metadata,
                "promoted": promoted,
                "warnings": warnings,
            },
        )

    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={"run_dir": str(run_dir)},
        )
        logger.exception("Full build failed")
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
    index_summary = None
    metrics = None

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

        if payload.select_best_checkpoint:
            update_status(phase="select-best-checkpoint")

            with log_stage(logger, "select-best-checkpoint", run_dir=str(run_dir)):
                best_summary = select_best_checkpoint(
                    run_dir=run_dir,
                    train_limit=payload.train_limit,
                    val_limit=payload.val_limit,
                    test_limit=payload.test_limit,
                    eval_query_limit=payload.eval_query_limit,
                    index_windows_override=payload.index_windows_override or 64,
                    use_full_query_audio=payload.use_full_query_audio,
                    fixed_eval_set_name=payload.fixed_eval_set_name,
                    eval_noise_mode=payload.best_checkpoint_eval_noise_mode,
                    metric_name=payload.best_checkpoint_metric,
                )

            model_path = run_dir / "model_best.pt"
        else:
            best_summary = None

        update_status(phase="evaluate")
        metrics_path = run_dir / "metrics.json"
        eval_results_path = run_dir / "eval_results.jsonl"
        with log_stage(logger, "evaluate-prepared", metrics_path=str(metrics_path)):
            metrics = evaluate_prepared(
                model_path=model_path,
                output_metrics_path=metrics_path,
                train_limit=payload.train_limit,
                val_limit=payload.val_limit,
                test_limit=payload.test_limit,
                eval_query_limit=payload.eval_query_limit,
                index_windows_override=payload.index_windows_override,
                use_full_query_audio=payload.use_full_query_audio,
                fixed_eval_set_name=payload.fixed_eval_set_name,
                eval_noise_mode=payload.eval_noise_mode,
                results_jsonl_path=eval_results_path,
                aggregation_strategies=payload.aggregation_strategies,
            )

        if payload.build_artifacts:
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
            index_meta_path = run_dir / "faiss_meta.json"

            with log_stage(logger, "build-faiss-index"):
                index_summary = build_faiss_index(
                    embeddings_path=embeddings_path,
                    song_ids_path=song_ids_path,
                    index_out_path=index_path,
                    meta_out_path=index_meta_path,
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
                "best_summary": best_summary,
                "index_summary": index_summary,
                "metrics": metrics,
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


def run_incremental_pipeline(payload: IncrementalSyncRequest) -> None:
    global pipeline_in_progress
    if pipeline_in_progress:
        return

    pipeline_in_progress = True
    run_dir = make_run_dir("incremental")

    update_status(
        current_job="incremental-sync",
        phase="detect-new-tracks",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={
            "run_dir": str(run_dir),
            "bucket": payload.bucket,
            "prefix": payload.prefix,
            "promote": payload.promote,
        },
    )
    reset_progress()

    try:
        update_status(phase="incremental-sync")

        with log_stage(logger, "incremental-sync", bucket=payload.bucket, prefix=payload.prefix):
            summary = build_incremental_run(
                bucket=payload.bucket,
                prefix=payload.prefix,
                run_dir=run_dir,
                max_new_tracks=payload.max_new_tracks,
                allow_incremental_when_retrain_recommended=payload.allow_incremental_when_retrain_recommended,
                index_windows_override=payload.index_windows_override,
            )

        if summary.get("no_changes"):
            update_status(
                phase="done",
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                progress={"run_dir": str(run_dir), "summary": summary},
            )
            return

        promoted = False

        if payload.promote:
            update_status(phase="promote")
            with log_stage(logger, "promote-active", run_dir=str(run_dir)):
                promote_run_to_active(run_dir)
            promoted = True

        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={
                "run_dir": str(run_dir),
                "summary": summary,
                "promoted": promoted,
                "warning": None if promoted else "Incremental run built but not promoted.",
            },
        )

    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={"run_dir": str(run_dir)},
        )
        logger.exception("Incremental sync failed")
    finally:
        pipeline_in_progress = False


@app.post("/pipeline/full-run", response_model=AsyncJobResponse)
async def full_run(payload: FullRunRequest, background_tasks: BackgroundTasks):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_full_pipeline, payload)
    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket=payload.bucket,
        prefix=payload.prefix,
    )


@app.post("/pipeline/experiment-run", response_model=AsyncJobResponse)
async def experiment_run(payload: ExperimentRunRequest, background_tasks: BackgroundTasks):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_experiment_pipeline, payload)
    return AsyncJobResponse(accepted=True, status="scheduled", bucket=payload.bucket, prefix=payload.prefix)


@app.post("/pipeline/incremental-sync", response_model=AsyncJobResponse)
async def incremental_sync(payload: IncrementalSyncRequest, background_tasks: BackgroundTasks):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_incremental_pipeline, payload)

    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket=payload.bucket,
        prefix=payload.prefix,
    )

@app.get("/pipeline/status")
async def pipeline_status():
    return {
        "pipeline_in_progress": pipeline_in_progress,
        "job_status": get_status(),
    }

@app.post("/pipeline/experiment/tune-thresholds")
async def tune_experiment_thresholds(payload: ThresholdTuneExperimentRequest):
    run_dir = Path(payload.run_dir)

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"run_dir does not exist: {run_dir}")

    results_path = run_dir / payload.results_filename

    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"eval results not found: {results_path}")

    output_path = run_dir / "threshold_tuning.json"

    try:
        summary = tune_thresholds(
            confidence_values=payload.confidence_values,
            margin_values=payload.margin_values,
            support_values=payload.support_values,
            results_path=str(results_path),
            output_path=str(output_path),
            case=payload.case,
            aggregation_strategy=payload.aggregation_strategy,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "run_dir": str(run_dir),
        "results_path": str(results_path),
        "output_path": str(output_path),
        "summary": summary,
    }

@app.post("/pipeline/evaluate-model")
async def evaluate_existing_model(payload: EvaluateExistingModelRequest):
    model_path = Path(payload.model_path)

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"model_path does not exist: {model_path}")

    run_dir = model_path.parent
    output_metrics_path = (
        Path(payload.output_metrics_path)
        if payload.output_metrics_path
        else run_dir / f"metrics_{model_path.stem}.json"
    )
    eval_results_path = run_dir / f"eval_results_{model_path.stem}.jsonl"

    try:
        metrics = evaluate_prepared(
            model_path=model_path,
            output_metrics_path=output_metrics_path,
            train_limit=payload.train_limit,
            val_limit=payload.val_limit,
            test_limit=payload.test_limit,
            eval_query_limit=payload.eval_query_limit,
            index_windows_override=payload.index_windows_override,
            use_full_query_audio=payload.use_full_query_audio,
            fixed_eval_set_name=payload.fixed_eval_set_name,
            eval_noise_mode=payload.eval_noise_mode,
            results_jsonl_path=eval_results_path,
            aggregation_strategies=payload.aggregation_strategies,
        )
    except Exception as exc:
        logger.exception(
            "Evaluate existing model failed model_path=%s output_metrics_path=%s",
            model_path,
            output_metrics_path,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "model_path": str(model_path),
        "output_metrics_path": str(output_metrics_path),
        "eval_results_path": str(eval_results_path),
        "metrics": metrics,
    }

@app.post("/pipeline/promote-run")
async def promote_run(payload: PromoteRunRequest):
    run_dir = Path(payload.run_dir)

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"run_dir does not exist: {run_dir}")

    try:
        validate_run_artifacts(run_dir, require_metrics=payload.require_metrics)
        promote_run_to_active(run_dir, require_metrics=payload.require_metrics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "promoted": True,
        "run_dir": str(run_dir),
        "require_metrics": payload.require_metrics,
    }

@app.get("/pipeline/retrain-status")
async def retrain_status_endpoint():
    try:
        return get_retrain_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/pipeline/cleanup")
async def cleanup_endpoint(payload: CleanupRequest):
    try:
        return cleanup_runs(
            dry_run=payload.dry_run,
            keep_last_full_runs=payload.keep_last_full_runs,
            keep_last_experiment_runs=payload.keep_last_experiment_runs,
            keep_last_incremental_runs=payload.keep_last_incremental_runs,
            delete_failed_runs=payload.delete_failed_runs,
            delete_epoch_checkpoints=payload.delete_epoch_checkpoints,
            delete_eval_results=payload.delete_eval_results,
            min_age_hours=payload.min_age_hours,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

def run_build_embeddings_only(payload: BuildEmbeddingsRequest) -> None:
    global pipeline_in_progress

    if pipeline_in_progress:
        return

    pipeline_in_progress = True

    run_dir = Path(payload.run_dir)
    model_path = Path(payload.model_path)

    embeddings_path = run_dir / payload.embeddings_filename
    song_ids_path = run_dir / payload.song_ids_filename
    manifest_path = run_dir / payload.manifest_filename

    update_status(
        current_job="build-embeddings-only",
        phase="validate",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "embeddings_path": str(embeddings_path),
            "song_ids_path": str(song_ids_path),
            "manifest_path": str(manifest_path),
            "index_all_prepared": payload.index_all_prepared,
            "index_windows_override": payload.index_windows_override,
        },
    )
    reset_progress()

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"model_path does not exist: {model_path}")

        run_dir.mkdir(parents=True, exist_ok=True)

        if payload.index_all_prepared:
            train_limit = 0
            val_limit = 0
            test_limit = 0
        else:
            train_limit = payload.train_limit
            val_limit = payload.val_limit
            test_limit = payload.test_limit

        update_status(phase="build-embeddings")

        with log_stage(logger, "build-embeddings-only", model_path=str(model_path)):
            summary = build_embeddings_from_prepared(
                model_path=model_path,
                embeddings_out=embeddings_path,
                song_ids_out=song_ids_path,
                manifest_out=manifest_path,
                train_limit=train_limit,
                val_limit=val_limit,
                test_limit=test_limit,
                index_windows_override=payload.index_windows_override,
            )

        summary["index_all_prepared"] = payload.index_all_prepared

        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={
                "run_dir": str(run_dir),
                "summary": summary,
                "embeddings_path": str(embeddings_path),
                "song_ids_path": str(song_ids_path),
                "manifest_path": str(manifest_path),
            },
        )

    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={
                "run_dir": str(run_dir),
                "model_path": str(model_path),
            },
        )
        logger.exception("Build embeddings only failed")

    finally:
        pipeline_in_progress = False

@app.post("/pipeline/build-embeddings-only", response_model=AsyncJobResponse)
async def build_embeddings_only_endpoint(
    payload: BuildEmbeddingsRequest,
    background_tasks: BackgroundTasks,
):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_build_embeddings_only, payload)

    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket="",
        prefix="",
    )

def run_build_faiss_only(payload: BuildFaissOnlyRequest) -> None:
    global pipeline_in_progress

    if pipeline_in_progress:
        return

    pipeline_in_progress = True

    run_dir = Path(payload.run_dir)

    embeddings_path = run_dir / payload.embeddings_filename
    song_ids_path = run_dir / payload.song_ids_filename
    index_path = run_dir / payload.index_filename
    meta_path = run_dir / payload.meta_filename

    update_status(
        current_job="build-faiss-only",
        phase="validate",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={
            "run_dir": str(run_dir),
            "embeddings_path": str(embeddings_path),
            "song_ids_path": str(song_ids_path),
            "index_path": str(index_path),
            "meta_path": str(meta_path),
            "index_type": payload.index_type,
            "nlist": payload.nlist,
            "nprobe": payload.nprobe,
        },
    )
    reset_progress()

    try:
        if not embeddings_path.exists():
            raise FileNotFoundError(f"embeddings file does not exist: {embeddings_path}")

        if not song_ids_path.exists():
            raise FileNotFoundError(f"song_ids file does not exist: {song_ids_path}")

        update_status(phase="build-index")

        with log_stage(logger, "build-faiss-only", embeddings_path=str(embeddings_path)):
            summary = build_faiss_index(
                embeddings_path=embeddings_path,
                song_ids_path=song_ids_path,
                index_out_path=index_path,
                meta_out_path=meta_path,
                index_type=payload.index_type,
                nlist=payload.nlist,
                nprobe=payload.nprobe,
            )

        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={
                "run_dir": str(run_dir),
                "summary": summary,
                "index_path": str(index_path),
                "meta_path": str(meta_path),
            },
        )

    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={
                "run_dir": str(run_dir),
                "embeddings_path": str(embeddings_path),
                "song_ids_path": str(song_ids_path),
            },
        )
        logger.exception("Build FAISS only failed")

    finally:
        pipeline_in_progress = False

@app.post("/pipeline/build-faiss-only", response_model=AsyncJobResponse)
async def build_faiss_only_endpoint(
    payload: BuildFaissOnlyRequest,
    background_tasks: BackgroundTasks,
):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_build_faiss_only, payload)

    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket="",
        prefix="",
    )

def run_evaluate_artifacts(payload: EvaluateArtifactsRequest) -> None:
    global pipeline_in_progress

    if pipeline_in_progress:
        return

    pipeline_in_progress = True

    model_path = Path(payload.model_path)
    index_path = Path(payload.index_path)
    song_ids_path = Path(payload.song_ids_path)
    output_metrics_path = Path(payload.output_metrics_path)
    output_results_path = Path(payload.output_results_path) if payload.output_results_path else None
    faiss_meta_path = Path(payload.faiss_meta_path) if payload.faiss_meta_path else None

    update_status(
        current_job="evaluate-artifacts",
        phase="validate",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={
            "model_path": str(model_path),
            "index_path": str(index_path),
            "song_ids_path": str(song_ids_path),
            "faiss_meta_path": str(faiss_meta_path) if faiss_meta_path else None,
            "output_metrics_path": str(output_metrics_path),
            "output_results_path": str(output_results_path) if output_results_path else None,
            "fixed_eval_set_name": payload.fixed_eval_set_name,
            "eval_noise_mode": payload.eval_noise_mode,
            "aggregation_strategies": payload.aggregation_strategies,
            "eval_query_limit": payload.eval_query_limit,
        },
    )
    reset_progress()

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"model_path does not exist: {model_path}")

        if not index_path.exists():
            raise FileNotFoundError(f"index_path does not exist: {index_path}")

        if not song_ids_path.exists():
            raise FileNotFoundError(f"song_ids_path does not exist: {song_ids_path}")

        update_status(phase="evaluate")

        with log_stage(logger, "evaluate-artifacts", model_path=str(model_path), index_path=str(index_path)):
            metrics = evaluate_artifacts(
                model_path=model_path,
                index_path=index_path,
                song_ids_path=song_ids_path,
                faiss_meta_path=faiss_meta_path,
                output_metrics_path=output_metrics_path,
                output_results_path=output_results_path,
                train_limit=payload.train_limit,
                val_limit=payload.val_limit,
                test_limit=payload.test_limit,
                eval_query_limit=payload.eval_query_limit,
                use_full_query_audio=payload.use_full_query_audio,
                fixed_eval_set_name=payload.fixed_eval_set_name,
                eval_noise_mode=payload.eval_noise_mode,
                aggregation_strategies=payload.aggregation_strategies,
            )

        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={
                "model_path": str(model_path),
                "index_path": str(index_path),
                "song_ids_path": str(song_ids_path),
                "output_metrics_path": str(output_metrics_path),
                "output_results_path": str(output_results_path) if output_results_path else str(metrics.get("eval_results_path")),
                "metrics": metrics,
            },
        )

    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={
                "model_path": str(model_path),
                "index_path": str(index_path),
                "song_ids_path": str(song_ids_path),
            },
        )
        logger.exception("Evaluate artifacts failed")

    finally:
        pipeline_in_progress = False

@app.post("/pipeline/evaluate-artifacts", response_model=AsyncJobResponse)
async def evaluate_artifacts_endpoint(
    payload: EvaluateArtifactsRequest,
    background_tasks: BackgroundTasks,
):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_evaluate_artifacts, payload)

    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket="",
        prefix="",
    )

def run_evaluate_reranker(payload: EvaluateRerankerRequest) -> None:
    global pipeline_in_progress

    if pipeline_in_progress:
        return

    pipeline_in_progress = True

    output_metrics_path = Path(payload.output_metrics_path)
    output_results_path = Path(payload.output_results_path)

    update_status(
        current_job="evaluate-reranker",
        phase="validate",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={
            "output_metrics_path": str(output_metrics_path),
            "output_results_path": str(output_results_path),
            "val_limit": payload.val_limit,
            "test_limit": payload.test_limit,
            "query_limit": payload.query_limit,
            "top_k": payload.top_k,
            "reranker_max_candidates": payload.reranker_max_candidates,
            "cases": payload.cases,
            "segment_seconds": payload.segment_seconds,
            "fp_weight": payload.fp_weight,
            "ml_weight": payload.ml_weight,
            "timeout_sec": payload.timeout_sec,
        },
    )
    reset_progress()

    try:
        update_status(phase="evaluate")

        with log_stage(
            logger,
            "evaluate-reranker",
            output_metrics_path=str(output_metrics_path),
            output_results_path=str(output_results_path),
        ):
            metrics = evaluate_reranker(
                output_metrics_path=output_metrics_path,
                output_results_path=output_results_path,
                val_limit=payload.val_limit,
                test_limit=payload.test_limit,
                query_limit=payload.query_limit,
                top_k=payload.top_k,
                reranker_max_candidates=payload.reranker_max_candidates,
                cases=payload.cases,
                segment_seconds=payload.segment_seconds,
                fp_weight=payload.fp_weight,
                ml_weight=payload.ml_weight,
                timeout_sec=payload.timeout_sec,
            )

        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={
                "output_metrics_path": str(output_metrics_path),
                "output_results_path": str(output_results_path),
                "metrics": metrics,
            },
        )

    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={
                "output_metrics_path": str(output_metrics_path),
                "output_results_path": str(output_results_path),
            },
        )
        logger.exception("Evaluate reranker failed")

    finally:
        pipeline_in_progress = False

@app.post("/pipeline/evaluate-reranker", response_model=AsyncJobResponse)
async def evaluate_reranker_endpoint(
    payload: EvaluateRerankerRequest,
    background_tasks: BackgroundTasks,
):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_evaluate_reranker, payload)

    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket="",
        prefix="",
    )

def run_build_reranker_reference_embeddings(payload: BuildRerankerReferenceEmbeddingsRequest) -> None:
    global pipeline_in_progress

    if pipeline_in_progress:
        return

    pipeline_in_progress = True

    run_dir = Path(payload.run_dir)
    model_path = Path(payload.model_path)

    embeddings_path = run_dir / payload.embeddings_filename
    meta_path = run_dir / payload.meta_filename
    config_path = run_dir / payload.config_filename

    update_status(
        current_job="build-reranker-reference-embeddings",
        phase="validate",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        last_error=None,
        progress={
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "embeddings_path": str(embeddings_path),
            "meta_path": str(meta_path),
            "config_path": str(config_path),
            "index_all_prepared": payload.index_all_prepared,
            "train_limit": payload.train_limit,
            "val_limit": payload.val_limit,
            "test_limit": payload.test_limit,
            "window_seconds": payload.window_seconds,
            "hop_seconds": payload.hop_seconds,
            "batch_size": payload.batch_size,
        },
    )
    reset_progress()

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"model_path does not exist: {model_path}")

        run_dir.mkdir(parents=True, exist_ok=True)

        update_status(phase="build-reranker-reference-embeddings")

        with log_stage(
            logger,
            "build-reranker-reference-embeddings",
            model_path=str(model_path),
            embeddings_path=str(embeddings_path),
            meta_path=str(meta_path),
            config_path=str(config_path),
        ):
            summary = build_reranker_reference_embeddings(
                model_path=model_path,
                embeddings_out=embeddings_path,
                meta_out=meta_path,
                config_out=config_path,
                train_limit=payload.train_limit,
                val_limit=payload.val_limit,
                test_limit=payload.test_limit,
                index_all_prepared=payload.index_all_prepared,
                window_seconds=payload.window_seconds,
                hop_seconds=payload.hop_seconds,
                batch_size=payload.batch_size,
            )

        update_status(
            phase="done",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_success=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress={
                "run_dir": str(run_dir),
                "model_path": str(model_path),
                "embeddings_path": str(embeddings_path),
                "meta_path": str(meta_path),
                "config_path": str(config_path),
                "summary": summary,
            },
        )

    except Exception as exc:
        update_status(
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=str(exc),
            progress={
                "run_dir": str(run_dir),
                "model_path": str(model_path),
            },
        )
        logger.exception("Build reranker reference embeddings failed")

    finally:
        pipeline_in_progress = False


@app.post("/pipeline/build-reranker-reference-embeddings", response_model=AsyncJobResponse)
async def build_reranker_reference_embeddings_endpoint(
    payload: BuildRerankerReferenceEmbeddingsRequest,
    background_tasks: BackgroundTasks,
):
    if pipeline_in_progress:
        raise HTTPException(status_code=409, detail="Pipeline job is already running")

    background_tasks.add_task(run_build_reranker_reference_embeddings, payload)

    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket="",
        prefix="",
    )

@app.get("/health")
async def health():
    return {
        "status": "busy" if pipeline_in_progress else "ok",
        "pipeline_in_progress": pipeline_in_progress,
    }
