import zipfile
import boto3
from botocore.exceptions import ClientError
from pathlib import PurePosixPath

S3_ENDPOINT = "http://localhost:9000"
S3_ACCESS_KEY = "minio"
S3_SECRET_KEY = "minio123"
S3_BUCKET = "tracks"
MAX_FILES = 5000

ZIP_PATH = "D:\\Кирилл\\Downloads\\fma_small.zip"

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)


def is_audio_file(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXT)


def make_s3_key(zip_path: str) -> str:
    p = PurePosixPath(zip_path)

    if len(p.parts) < 2:
        return None

    folder = p.parts[-2]
    filename = p.parts[-1]

    if not folder.isdigit():
        return None

    return f"fma/{folder}/{filename}"

def object_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def upload_fma_zip(zip_path):
    uploaded = 0
    skipped = 0

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():

            if info.is_dir():
                continue

            if not is_audio_file(info.filename):
                continue

            s3_key = make_s3_key(info.filename)
            if not s3_key:
                continue

            if object_exists(S3_BUCKET, s3_key):
                skipped += 1
                continue

            with zf.open(info) as file_stream:
                s3.upload_fileobj(file_stream, S3_BUCKET, s3_key)

            uploaded += 1
            if uploaded % 100 == 0:
                print(f"Uploaded {uploaded}, skipped {skipped}")
            if uploaded >= MAX_FILES:
                break

    print("Done. Uploaded:", uploaded)


if __name__ == "__main__":
    upload_fma_zip(ZIP_PATH)
