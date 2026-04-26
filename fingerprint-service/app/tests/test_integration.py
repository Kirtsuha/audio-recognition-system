import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from main import app

def make_wav():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(44100), 44100, format="WAV")
    buf.seek(0)
    buf.name = "test.wav"
    return buf

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

@pytest.mark.integration
def test_index_and_s3_pipeline(client, s3_client, db):
    # ---------------- Загружаем тестовый аудиофайл в S3 ----------------
    buf = make_wav()
    audio_content = buf.read()  # "молчаливый" WAV
    s3_key = "test_integration/test.wav"
    S3_BUCKET = "tracks"
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=audio_content)

    # ---------------- Проверка /index до индексирования ----------------
    fake_file = io.BytesIO(audio_content)
    fake_file.name = "test.wav"
    files = {"file": ("test.wav", fake_file, "audio/wav")}
    response = client.post("/index", files=files)
    assert response.status_code == 200
    assert response.json()["match"] is False  # ещё не индексировали

    # ---------------- Индексируем трек через pipeline ----------------
    from service.postgres_loader_pipeline import process_s3_track
    process_s3_track(s3_key, db)

    # ---------------- Проверка /index после индексирования ----------------
    fake_file = io.BytesIO(audio_content)
    fake_file.name = "test.wav"
    files = {"file": ("test.wav", fake_file, "audio/wav")}
    response = client.post("/index", files=files)
    assert response.status_code == 200

    assert response.json()["match"] is True
    assert "track_id" in response.json()
    assert "confidence" in response.json()