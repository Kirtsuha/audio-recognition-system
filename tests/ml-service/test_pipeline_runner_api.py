from fastapi.testclient import TestClient

import pipeline_runner.main as runner


client = TestClient(runner.app)


def test_pipeline_health_ok(monkeypatch):
    monkeypatch.setattr(runner, "pipeline_in_progress", False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_pipeline_health_busy(monkeypatch):
    monkeypatch.setattr(runner, "pipeline_in_progress", True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "busy"


def test_pipeline_status(monkeypatch):
    monkeypatch.setattr(runner, "pipeline_in_progress", False)
    monkeypatch.setattr(runner, "get_status", lambda: {"phase": "done"})

    response = client.get("/pipeline/status")

    assert response.status_code == 200
    assert response.json()["pipeline_in_progress"] is False
    assert response.json()["job_status"]["phase"] == "done"


def test_full_run_scheduled(monkeypatch):
    monkeypatch.setattr(runner, "pipeline_in_progress", False)
    monkeypatch.setattr(runner, "run_full_pipeline", lambda bucket, prefix: None)

    response = client.post(
        "/pipeline/full-run",
        json={"bucket": "tracks", "prefix": "fma/"},
    )

    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert response.json()["status"] == "scheduled"


def test_full_run_conflict(monkeypatch):
    monkeypatch.setattr(runner, "pipeline_in_progress", True)

    response = client.post(
        "/pipeline/full-run",
        json={"bucket": "tracks", "prefix": "fma/"},
    )

    assert response.status_code == 409


def test_experiment_run_scheduled(monkeypatch):
    monkeypatch.setattr(runner, "pipeline_in_progress", False)
    monkeypatch.setattr(runner, "run_experiment_pipeline", lambda payload: None)

    response = client.post(
        "/pipeline/experiment-run",
        json={
            "bucket": "tracks",
            "prefix": "fma/",
            "train_limit": 10,
            "val_limit": 5,
            "test_limit": 5,
            "prepare_limit": 20,
            "skip_prepare": True,
            "epochs_override": 1,
            "eval_query_limit": 5,
            "index_windows_override": 2,
            "use_full_query_audio": False,
            "experiment_name": "test-exp",
        },
    )

    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_incremental_sync_scheduled(monkeypatch):
    monkeypatch.setattr(runner, "pipeline_in_progress", False)
    monkeypatch.setattr(runner, "run_incremental_pipeline", lambda bucket, prefix: None)

    response = client.post(
        "/pipeline/incremental-sync",
        json={"bucket": "tracks", "prefix": "fma/"},
    )

    assert response.status_code == 200
    assert response.json()["accepted"] is True