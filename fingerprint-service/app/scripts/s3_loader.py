import zipfile
import boto3
import os

S3_ENDPOINT = "http://localhost:9000"
S3_ACCESS_KEY = "minio"
S3_SECRET_KEY = "minio123"
S3_BUCKET = "tracks"

ZIP_PATH = "dataset.zip"

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

def upload_zip_streaming(zip_path):
    s3_mapping = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith((".wav", ".mp3", ".flac")):
                with zf.open(info) as file_stream:
                    s3_key = os.path.basename(info.filename)
                    s3.upload_fileobj(file_stream, S3_BUCKET, s3_key)
                    s3_mapping[info.filename] = s3_key
                    print(f"Uploaded {info.filename} → {s3_key}")
    return s3_mapping

if __name__ == "__main__":
    mapping = upload_zip_streaming(ZIP_PATH)
    print("Uploaded files:", mapping)
