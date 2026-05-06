import csv
import os
import re
import zipfile
from pathlib import PurePosixPath
from urllib.parse import quote, unquote

from repository.s3_client import ensure_bucket_exists, get_s3, object_exists


MAX_FILES = int(os.getenv("S3_LOADER_MAX_FILES", "5000"))
S3_BUCKET = os.getenv("S3_BUCKET", "tracks")
AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


def is_audio_file(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXT)


def _clean_segment(value: str, fallback: str) -> str:
    cleaned = _CONTROL_CHARS.sub("", str(value or "")).strip()
    cleaned = cleaned.replace("\\", "_").replace("/", "_")
    cleaned = _WHITESPACE.sub(" ", cleaned).strip(" .")
    return cleaned or fallback


def make_track_s3_key(artist: str, title: str, extension: str = ".mp3", prefix: str = "") -> str:
    safe_artist = _clean_segment(artist, "unknown_artist")
    safe_title = _clean_segment(title, "unknown_title")
    safe_ext = extension if extension.startswith(".") else f".{extension}"
    safe_ext = safe_ext.lower()

    key = f"{safe_artist}/{safe_title}{safe_ext}"
    prefix = prefix.strip("/")
    return f"{prefix}/{key}" if prefix else key


def encode_metadata(title: str, artist: str, album: str | None = None) -> dict:
    metadata = {
        "title": quote(str(title), safe=""),
        "artist": quote(str(artist), safe=""),
    }

    if album:
        metadata["album"] = quote(str(album), safe="")

    return metadata


def decode_metadata(metadata: dict | None) -> tuple[str | None, str | None]:
    metadata = metadata or {}
    title = metadata.get("title")
    artist = metadata.get("artist")
    return (
        unquote(title) if title else None,
        unquote(artist) if artist else None,
    )


def _read_manifest(manifest_path: str) -> tuple[dict[str, dict], dict]:
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as fileobj:
        reader = csv.DictReader(fileobj)
        required = {"path", "title", "artist"}
        missing = required.difference(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")

        rows = {}
        stats = {
            "manifest_rows": 0,
            "manifest_skipped_empty_path": 0,
            "manifest_unknown_title": 0,
            "manifest_unknown_artist": 0,
        }

        for line_number, row in enumerate(reader, start=2):
            stats["manifest_rows"] += 1
            source_path = str(row.get("path") or "").strip()
            title = str(row.get("title") or "").strip()
            artist = str(row.get("artist") or "").strip()

            if not source_path:
                stats["manifest_skipped_empty_path"] += 1
                continue

            if not title:
                title = "UNKNOWN"
                stats["manifest_unknown_title"] += 1

            if not artist:
                artist = "UNKNOWN"
                stats["manifest_unknown_artist"] += 1

            normalized_path = str(PurePosixPath(source_path.replace("\\", "/")))

            if normalized_path in rows:
                raise ValueError(f"Duplicate manifest path: {normalized_path}")

            rows[normalized_path] = {
                "path": normalized_path,
                "title": title,
                "artist": artist,
            }

        return rows, stats


def upload_tracks_zip(
    zip_path: str,
    manifest_path: str,
    bucket: str | None = None,
    prefix: str = "",
    max_files: int | None = None,
) -> dict:
    s3 = get_s3()
    bucket_name = bucket or S3_BUCKET
    ensure_bucket_exists(bucket_name)

    manifest, manifest_stats = _read_manifest(manifest_path)
    upload_limit = MAX_FILES if max_files is None else max_files

    uploaded = 0
    skipped = 0
    ignored = 0
    failed = 0
    missing = 0
    uploaded_keys = []

    with zipfile.ZipFile(zip_path) as archive:
        archive_files = {
            str(PurePosixPath(info.filename.replace("\\", "/"))): info
            for info in archive.infolist()
            if not info.is_dir()
        }

        for source_path, row in manifest.items():
            if upload_limit and uploaded >= upload_limit:
                break

            info = archive_files.get(source_path)

            if info is None:
                missing += 1
                continue

            if not is_audio_file(info.filename):
                ignored += 1
                continue

            extension = PurePosixPath(info.filename).suffix or ".mp3"
            s3_key = make_track_s3_key(
                artist=row["artist"],
                title=row["title"],
                extension=extension,
                prefix=prefix,
            )

            if object_exists(s3_key, bucket_name):
                skipped += 1
                continue

            try:
                with archive.open(info) as file_stream:
                    s3.upload_fileobj(
                        file_stream,
                        bucket_name,
                        s3_key,
                        ExtraArgs={"Metadata": encode_metadata(row["title"], row["artist"])},
                    )

                uploaded += 1
                uploaded_keys.append(s3_key)

            except Exception:
                failed += 1

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "ignored": ignored,
        "missing": missing,
        "failed": failed,
        **manifest_stats,
        "bucket": bucket_name,
        "prefix": prefix,
        "keys": uploaded_keys,
    }


def upload_single_track(
    fileobj,
    filename: str,
    title: str,
    artist: str,
    album: str | None = None,
    bucket: str | None = None,
    prefix: str = "",
    overwrite: bool = False,
) -> dict:
    if not title or not artist:
        raise ValueError("title and artist are required")

    if not is_audio_file(filename):
        raise ValueError("Unsupported audio format")

    s3 = get_s3()
    bucket_name = bucket or S3_BUCKET
    ensure_bucket_exists(bucket_name)

    extension = PurePosixPath(filename).suffix or ".mp3"
    s3_key = make_track_s3_key(
        artist=artist,
        title=title,
        extension=extension,
        prefix=prefix,
    )

    if object_exists(s3_key, bucket_name) and not overwrite:
        return {
            "uploaded": 0,
            "skipped": 1,
            "bucket": bucket_name,
            "s3_key": s3_key,
            "reason": "already_exists",
        }

    s3.upload_fileobj(
        fileobj,
        bucket_name,
        s3_key,
        ExtraArgs={"Metadata": encode_metadata(title, artist, album)},
    )

    return {
        "uploaded": 1,
        "skipped": 0,
        "bucket": bucket_name,
        "s3_key": s3_key,
        "title": title,
        "artist": artist,
        "album": album,
    }


def upload_fma_zip(zip_path: str, max_files: int | None = None) -> dict:
    raise ValueError("upload_fma_zip is deprecated; use upload_tracks_zip with manifest.csv")


if __name__ == "__main__":
    zip_path = os.getenv("ZIP_PATH")
    manifest_path = os.getenv("MANIFEST_PATH")
    if not zip_path or not manifest_path:
        raise ValueError("ZIP_PATH and MANIFEST_PATH environment variables are required")
    upload_tracks_zip(zip_path, manifest_path)
