import os
import boto3
from botocore.exceptions import ClientError

def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        region_name="us-east-1",
        use_ssl=False
    )


def ensure_bucket_exists(bucket: str | None = None):
    s3 = get_s3()
    bucket_name = bucket or os.getenv("S3_BUCKET")
    if not bucket_name:
        raise ValueError("S3_BUCKET is not configured")

    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError:
        s3.create_bucket(Bucket=bucket_name)


def object_exists(key: str, bucket: str | None = None) -> bool:
    s3 = get_s3()
    bucket_name = bucket or os.getenv("S3_BUCKET")
    if not bucket_name:
        raise ValueError("S3_BUCKET is not configured")

    try:
        s3.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            return False
        raise


def upload_fileobj(fileobj, key):
    s3 = get_s3()
    bucket = os.getenv("S3_BUCKET")
    s3.upload_fileobj(fileobj, bucket, key)
