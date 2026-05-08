import os
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

FINGERPRINT_APP_DIR = Path(__file__).resolve().parents[2] / "fingerprint-service" / "app"
