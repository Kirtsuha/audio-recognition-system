import boto3
import tempfile

from s3.s3_client import s3

def download_song(bucket, key):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    s3.download_file(bucket, key, tmp.name)
    return tmp.name