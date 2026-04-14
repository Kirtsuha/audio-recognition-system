import os
import boto3

def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        region_name="us-east-1",
        use_ssl=False
    )

def upload_fileobj(fileobj, key):
    s3 = get_s3()
    bucket = os.getenv("S3_BUCKET")
    s3.upload_fileobj(fileobj, bucket, key)
