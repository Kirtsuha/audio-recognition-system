from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_health_degraded(monkeypatch):
    monkeypatch.setattr(main, "runtime_status", lambda: {"runtime_ready": False, "x": 1})

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["runtime_ready"] is False


def test_admin_status(monkeypatch):
    monkeypatch.setattr(main, "runtime_status", lambda: {"runtime_ready": True})
    monkeypatch.setattr(main, "load_status_from_disk", lambda: {"phase": "done"})

    response = client.get("/admin/status")

    assert response.status_code == 200
    assert response.json()["runtime_ready"] is True
    assert response.json()["job_status"]["phase"] == "done"


def test_reload_runtime_success(monkeypatch):
    called = {"value": False}

    def fake_load():
        called["value"] = True

    monkeypatch.setattr(main, "load_runtime_artifacts", fake_load)
    monkeypatch.setattr(main, "runtime_ready", lambda: True)

    response = client.post("/admin/reload-runtime")

    assert response.status_code == 200
    assert response.json() == {"reloaded": True, "runtime_ready": True}
    assert called["value"] is True


def test_reload_runtime_failure(monkeypatch):
    def fake_load():
        raise RuntimeError("broken artifacts")

    monkeypatch.setattr(main, "load_runtime_artifacts", fake_load)

    response = client.post("/admin/reload-runtime")

    assert response.status_code == 500
    assert "broken artifacts" in response.json()["detail"]


def test_recognize_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "recognize_audio_bytes",
        lambda audio: {
            "matched": True,
            "song_id": 10,
            "confidence": 0.9,
            "margin": 0.5,
            "support": 3,
            "top_candidates": [],
        },
    )

    response = client.post(
        "/recognize",
        files={"file": ("query.wav", b"fake-audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["matched"] is True
    assert response.json()["song_id"] == 10


def test_recognition_resolve_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "recognize_audio_bytes",
        lambda audio: {
            "matched": False,
            "reason": "low_confidence",
            "top_candidates": [],
        },
    )

    response = client.post(
        "/recognition/resolve",
        files={"file": ("query.wav", b"fake-audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["matched"] is False
    assert response.json()["reason"] == "low_confidence"


def test_recognize_runtime_error(monkeypatch):
    def fake_recognize(audio):
        raise RuntimeError("not ready")

    monkeypatch.setattr(main, "recognize_audio_bytes", fake_recognize)

    response = client.post(
        "/recognize",
        files={"file": ("query.wav", b"fake-audio", "audio/wav")},
    )

    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]


def test_recognize_unknown_error(monkeypatch):
    def fake_recognize(audio):
        raise ValueError("bad audio")

    monkeypatch.setattr(main, "recognize_audio_bytes", fake_recognize)

    response = client.post(
        "/recognize",
        files={"file": ("query.wav", b"fake-audio", "audio/wav")},
    )

    assert response.status_code == 500
    assert "bad audio" in response.json()["detail"]