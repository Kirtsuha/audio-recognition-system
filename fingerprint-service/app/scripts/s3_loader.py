import os
import zipfile
from pathlib import PurePosixPath

from repository.s3_client import ensure_bucket_exists, get_s3, object_exists


MAX_FILES = int(os.getenv("S3_LOADER_MAX_FILES", "5000"))
S3_BUCKET = os.getenv("S3_BUCKET", "tracks")
S3_PREFIX = os.getenv("S3_PREFIX", "fma/").rstrip("/") + "/"
AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg")


def is_audio_file(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXT)


def make_s3_key(zip_member_path: str) -> str | None:
    parts = [part for part in PurePosixPath(zip_member_path).parts if part not in {"", "."}]
    if len(parts) < 2:
        return None

    folder = parts[-2]
    filename = parts[-1]

    if not filename.lower().endswith(AUDIO_EXT):
        return None

    return f"{S3_PREFIX}{folder}/{filename}"


def upload_fma_zip(zip_path: str, max_files: int | None = None) -> dict:
    s3 = get_s3()
    ensure_bucket_exists(S3_BUCKET)

    upload_limit = MAX_FILES if max_files is None else max_files
    uploaded = 0
    skipped = 0
    ignored = 0

    print("Starting ZIP upload to S3 from %s", zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            if not is_audio_file(info.filename):
                ignored += 1
                continue

            s3_key = make_s3_key(info.filename)
            if not s3_key:
                ignored += 1
                print("Skipping file with unsupported archive layout: %s", info.filename)
                continue

            if object_exists(s3_key, S3_BUCKET):
                skipped += 1
                print("Skipping existing track in S3: %s", s3_key)
                continue

            with archive.open(info) as file_stream:
                s3.upload_fileobj(file_stream, S3_BUCKET, s3_key)

            uploaded += 1
            print("Uploaded track to S3: %s", s3_key)

            if upload_limit and uploaded >= upload_limit:
                print("Upload limit reached: %s files", upload_limit)
                break

    summary = {
        "uploaded": uploaded,
        "skipped": skipped,
        "ignored": ignored,
        "bucket": S3_BUCKET,
        "prefix": S3_PREFIX,
    }
    print("ZIP upload finished: %s", summary)
    return summary


if __name__ == "__main__":
    zip_path = os.getenv("ZIP_PATH")
    if not zip_path:
        raise ValueError("ZIP_PATH environment variable is required")
    upload_fma_zip(zip_path)
