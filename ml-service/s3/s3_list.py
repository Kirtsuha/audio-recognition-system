from s3.s3_client import s3


def list_all_songs(bucket: str, prefix: str = "fma/") -> list[str]:
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
            if key.endswith(".mp3") or key.endswith(".wav"):
                keys.append(key)

        if response.get("IsTruncated"):
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return sorted(keys)