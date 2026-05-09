from __future__ import annotations

import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError, FlexibleChecksumError

from app.config import settings

logger = logging.getLogger("ml-reranker.s3")


class S3Storage:
    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        )

    def get_bytes(self, bucket: str, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(f"S3 object not found {bucket}/{key}") from exc

            logger.exception("Failed to read S3 object bucket=%s key=%s", bucket, key)
            raise RuntimeError(f"Failed to read S3 object {bucket}/{key}: {exc}") from exc
        except (FlexibleChecksumError, BotoCoreError, OSError) as exc:
            logger.exception("Failed to read S3 object bucket=%s key=%s", bucket, key)
            raise RuntimeError(f"Failed to read S3 object {bucket}/{key}: {exc}") from exc

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self.client.head_bucket(Bucket=bucket)
        except ClientError:
            self.client.create_bucket(Bucket=bucket)
