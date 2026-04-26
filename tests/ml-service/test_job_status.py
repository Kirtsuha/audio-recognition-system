import pipeline.job_status as job_status


def test_job_status_update_progress_reset_and_load(tmp_path, monkeypatch):
    status_path = tmp_path / "job_status.json"
    monkeypatch.setattr(job_status, "JOB_STATUS_PATH", status_path)

    job_status.update_status(
        current_job="test",
        phase="running",
        last_error=None,
    )
    job_status.update_progress(done=1, total=10)

    current = job_status.get_status()
    assert current["current_job"] == "test"
    assert current["phase"] == "running"
    assert current["progress"]["done"] == 1

    loaded = job_status.load_status_from_disk()
    assert loaded["current_job"] == "test"

    reset = job_status.reset_progress()
    assert reset["progress"] == {}