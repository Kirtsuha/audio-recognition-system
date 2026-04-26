import logging
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

from repository.s3_client import ensure_bucket_exists, get_s3, object_exists


logger = logging.getLogger(__name__)

DATASET_NAME = os.getenv(
    "HF_FMA_DATASET",
    "benjamin-paine/free-music-archive-full",
)

S3_BUCKET = os.getenv("S3_FULL_BUCKET", "full-tracks")
S3_PREFIX = os.getenv("S3_FULL_PREFIX", "").strip("/")
if S3_PREFIX:
    S3_PREFIX = S3_PREFIX + "/"

import subprocess


def write_mp3(tmp_wav_path: str, tmp_mp3_path: str):
    cmd = [
        "ffmpeg",
        "-y",                 # overwrite
        "-loglevel", "error", # без спама
        "-i", tmp_wav_path,
        "-acodec", "libmp3lame",
        "-ab", "128k",        # битрейт (можно 64k / 192k)
        tmp_mp3_path,
    ]

    subprocess.run(cmd, check=True)

def make_s3_key(track_id: str, extension: str = ".mp3") -> str:
    filename = f"{str(track_id).zfill(6)}{extension}"
    folder = filename[:3]
    return f"{S3_PREFIX}{folder}/{filename}"

def decode_audio(audio_obj):
    if isinstance(audio_obj, dict):
        if audio_obj.get("array") is not None:
            return audio_obj["array"], audio_obj["sampling_rate"]

    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()

        data = samples.data
        sample_rate = samples.sample_rate

        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()

        data = np.asarray(data)

        if data.ndim == 2:
            data = data.T

        return data, sample_rate

    raise ValueError(f"Unsupported audio object type: {type(audio_obj)}")


def get_track_id(row: dict, index: int) -> str:
    for key in ("track_id", "id", "filename", "path"):
        value = row.get(key)
        if value is not None:
            stem = Path(str(value)).stem
            if stem:
                return stem

    return f"{index:06d}"


def get_audio(row: dict):
    audio = row.get("audio")
    if audio is None:
        return None

    if isinstance(audio, dict):
        if audio.get("array") is not None or audio.get("bytes") is not None:
            return audio

    if hasattr(audio, "get_all_samples"):
        return audio

    return None


def upload_hf_fma_to_s3(
    max_files: int,
    job_id: str | None = None,
    progress_callback=None,
) -> dict:
    job_id = job_id or str(uuid.uuid4())

    print(
        f"[{job_id}] Starting HF FMA upload: "
        f"dataset={DATASET_NAME}, max_files={max_files}, "
        f"bucket={S3_BUCKET}, prefix={S3_PREFIX}",
        flush=True,
    )

    s3 = get_s3()
    ensure_bucket_exists(S3_BUCKET)

    hf_token = os.getenv("HF_TOKEN")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
        token=hf_token,
    )

    uploaded = 0
    skipped = 0
    failed = 0
    scanned = 0

    def report(status: str, current_key: str | None = None, error: str | None = None):
        payload = {
            "job_id": job_id,
            "status": status,
            "dataset": DATASET_NAME,
            "max_files": max_files,
            "uploaded": uploaded,
            "skipped": skipped,
            "failed": failed,
            "scanned": scanned,
            "bucket": S3_BUCKET,
            "prefix": S3_PREFIX,
            "current_key": current_key,
            "error": error,
        }

        if progress_callback:
            progress_callback(payload)

        return payload

    report("running")

    for row in dataset:
        if uploaded >= max_files:
            break

        scanned += 1

        try:
            audio = get_audio(row)
            if audio is None:
                skipped += 1

                if scanned % 100 == 0:
                    print(
                        f"[{job_id}] Progress: "
                        f"scanned={scanned}, uploaded={uploaded}/{max_files}, "
                        f"skipped={skipped}, failed={failed}",
                        flush=True,
                    )
                    report("running")

                continue

            track_id = get_track_id(row, scanned)
            s3_key = make_s3_key(track_id)

            if object_exists(s3_key, S3_BUCKET):
                skipped += 1
                print(f"[{job_id}] Skipping existing object: {s3_key}", flush=True)
                report("running", current_key=s3_key)
                continue

            array, sampling_rate = decode_audio(audio)

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav, \
                    tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_mp3:

                sf.write(tmp_wav.name, array, sampling_rate)
                tmp_wav.flush()

                write_mp3(tmp_wav.name, tmp_mp3.name)

                with open(tmp_mp3.name, "rb") as fileobj:
                    s3.upload_fileobj(fileobj, S3_BUCKET, s3_key)

            uploaded += 1

            print(
                f"[{job_id}] Uploaded {uploaded}/{max_files}: {s3_key}",
                flush=True,
            )

            if uploaded % 25 == 0:
                report("running", current_key=s3_key)

        except Exception as exc:
            failed += 1

            print(
                f"[{job_id}] Failed on row {scanned}: {exc}",
                flush=True,
            )

            report("running", error=str(exc))

    summary = report("completed")

    print(
        f"[{job_id}] HF FMA upload completed: {summary}",
        flush=True,
    )

    return summary


if __name__ == "__main__":
    max_files = int(os.getenv("S3_LOADER_MAX_FILES", "8000"))
    upload_hf_fma_to_s3(max_files=max_files)