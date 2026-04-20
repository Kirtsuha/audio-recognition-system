import tempfile

from s3.s3_client import s3


def download_song(bucket: str, key: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    s3.download_file(bucket, key, tmp.name)
    return tmp.name