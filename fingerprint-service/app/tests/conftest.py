import pytest
import httpx
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
#from repository.database_config import Base

# ---------------- Тестовая БД (можно использовать ту же, что и для контейнера) ----------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fingerprint_user:fingerprint_pass@localhost:5432/fingerprint_db"
)
engine = create_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(bind=engine)

# ---------------- HTTP клиент для FastAPI ----------------
@pytest.fixture(scope="session")
def client():
    base_url = os.getenv("FINGERPRINT_URL", "http://localhost:8000")
    with httpx.Client(base_url=base_url) as c:
        yield c

# ---------------- Сессия БД для прямого взаимодействия ----------------
@pytest.fixture()
def db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- S3 клиент ----------------
@pytest.fixture()
def s3_client():
    import boto3
    S3_BUCKET = os.getenv("S3_BUCKET", "tracks")
    S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
    S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio123")

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="us-east-1",
        use_ssl=False
    )
    yield s3