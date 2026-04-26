import os
import tempfile
import time

from botocore.exceptions import BotoCoreError, ClientError, FlexibleChecksumError

from s3.s3_client import s3


def download_song(bucket: str, key: str, retries: int = 3, retry_delay_sec: float = 1.0) -> str:
    last_exc = None

    for attempt in range(1, retries + 1):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            s3.download_file(bucket, key, tmp_path)

            if not os.path.exists(tmp_path):
                raise FileNotFoundError(f"Downloaded file does not exist: {tmp_path}")

            if os.path.getsize(tmp_path) == 0:
                raise ValueError(f"Downloaded file is empty: bucket={bucket}, key={key}")

            return tmp_path

        except (FlexibleChecksumError, BotoCoreError, ClientError, OSError, ValueError) as exc:
            last_exc = exc
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            if attempt < retries:
                time.sleep(retry_delay_sec * attempt)
            else:
                raise RuntimeError(
                    f"Failed to download song from S3 after {retries} attempts: bucket={bucket}, key={key}"
                ) from exc

    raise RuntimeError(f"Unexpected S3 download failure: bucket={bucket}, key={key}") from last_exc