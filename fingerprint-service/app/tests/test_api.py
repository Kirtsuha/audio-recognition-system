import io
import numpy as np
import soundfile as sf

def make_wav():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(44100), 44100, format="WAV")
    buf.seek(0)
    buf.name = "test.wav"
    return buf

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_index_track_no_match(client):
    fake_audio = make_wav()
    response = client.post("/index", files={"file": ("test.wav", fake_audio, "audio/wav")})
    assert response.status_code == 200
    assert "match" in response.json()