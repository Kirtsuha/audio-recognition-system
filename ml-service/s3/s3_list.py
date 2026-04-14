from s3.s3_client import s3

def list_all_songs(bucket, prefix="fma/"):

    keys = []
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

            if key.endswith(".mp3"):
                keys.append(key)

        if response.get("IsTruncated"):
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return keys