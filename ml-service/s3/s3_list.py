from s3.s3_client import s3


SUPPORTED_AUDIO_SUFFIXES = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".webm",
}


def is_audio_key(key: str) -> bool:
    return any(key.lower().endswith(suffix) for suffix in SUPPORTED_AUDIO_SUFFIXES)


def list_all_songs(bucket: str, prefix: str = "") -> list[str]:
    keys: list[str] = []
    continuation_token = None

    while True:
        kwargs = {
            "Bucket": bucket,
            "Prefix": prefix,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**kwargs)

        for obj in response.get("Contents", []):
            key = obj["Key"]
            if is_audio_key(key):
                keys.append(key)

        if response.get("IsTruncated"):
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return sorted(keys)
