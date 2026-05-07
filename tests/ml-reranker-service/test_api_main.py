from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_health_degraded(monkeypatch):
    monkeypatch.setattr(main.engine, "status", lambda: main.RerankerEngine.status(main.engine))
    monkeypatch.setattr(main.engine, "model", None)
    monkeypatch.setattr(main.engine, "reference_store", None)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_reload_model_success(monkeypatch):
    monkeypatch.setattr(main.engine, "load_model", lambda: setattr(main.engine, "model", object()))

    response = client.post("/admin/reload-model")

    assert response.status_code == 200
    assert response.json()["reloaded"] is True


def test_rerank_runtime_error(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("not ready")

    monkeypatch.setattr(main.engine, "rerank", fail)

    response = client.post(
        "/rerank",
        json={
            "request_id": "r1",
            "query_audio": {"bucket": "q", "key": "a.wav"},
            "fingerprint": {"candidates": []},
        },
    )

    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]
