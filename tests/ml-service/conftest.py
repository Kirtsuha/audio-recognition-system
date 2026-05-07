import os
import sys
import types
from pathlib import Path

os.environ.setdefault("DISABLE_KAFKA_CONSUMER", "true")
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
os.environ.setdefault("S3_ENDPOINT", "http://localhost:9000")
os.environ.setdefault("S3_ACCESS_KEY", "test")
os.environ.setdefault("S3_SECRET_KEY", "test")
os.environ.setdefault("TRACKS_DATABASE_URL", "postgresql://test:test@localhost:5432/test")

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent.parent
ML_SERVICE_DIR = PROJECT_ROOT / "ml-service"

if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))


class FakeKafkaProducer:
    def __init__(self, *args, **kwargs):
        pass

    def send(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


class FakeKafkaConsumer:
    def __init__(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter(())


fake_kafka = types.ModuleType("kafka")
fake_kafka.KafkaProducer = FakeKafkaProducer
fake_kafka.KafkaConsumer = FakeKafkaConsumer
sys.modules["kafka"] = fake_kafka


fake_psycopg = types.ModuleType("psycopg")

def fake_connect(*args, **kwargs):
    raise RuntimeError("psycopg.connect should be mocked in tests")

fake_psycopg.connect = fake_connect
sys.modules["psycopg"] = fake_psycopg


class FakeMelSpectrogram:
    def __init__(self, *args, n_mels=128, **kwargs):
        self.n_mels = n_mels

    def to(self, device):
        return self

    def __call__(self, audio_batch):
        import torch

        batch_size = audio_batch.shape[0]
        frames = max(1, audio_batch.shape[-1] // 512)
        return torch.ones((batch_size, self.n_mels, frames), dtype=torch.float32, device=audio_batch.device)


fake_torchaudio = types.ModuleType("torchaudio")
fake_torchaudio.transforms = types.SimpleNamespace(MelSpectrogram=FakeMelSpectrogram)
sys.modules.setdefault("torchaudio", fake_torchaudio)
