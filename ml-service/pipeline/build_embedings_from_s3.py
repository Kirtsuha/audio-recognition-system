import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from app.model import AudioEncoder
from pipeline.config import (
    MODEL_PATH,
    EMBEDDINGS_PATH,
    SONG_IDS_PATH,
    SONG_MANIFEST_PATH,
    INDEX_WINDOWS_PER_SONG,
)
from pipeline.dataset import load_audio, extract_uniform_index_windows
from pipeline.to_mel import to_mel
from s3.s3_list import list_all_songs
from s3.s3_loader import download_song

logger = logging.getLogger("ml-service.embed-build")


def build_embeddings_from_s3(bucket: str, prefix: str) -> None:
    started_total = time.time()

    keys = list_all_songs(bucket=bucket, prefix=prefix)
    if not keys:
        raise ValueError("No songs found in S3")

    logger.info(
        "Embedding build started bucket=%s prefix=%s songs=%s windows_per_song=%s",
        bucket,
        prefix,
        len(keys),
        INDEX_WINDOWS_PER_SONG,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioEncoder().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_embeddings = []
    all_song_ids = []
    manifest = []

    skipped = 0
    processed = 0

    for idx, key in enumerate(keys, start=1):
        path = None
        try:
            path = download_song(bucket, key)

            if not os.path.exists(path):
                raise FileNotFoundError(f"Downloaded path does not exist: {path}")

            if os.path.getsize(path) == 0:
                raise ValueError(f"Downloaded file is empty: key={key}")

            audio = load_audio(path)
            if audio is None or len(audio) == 0:
                raise ValueError(f"Decoded empty audio for key={key}")

            windows = extract_uniform_index_windows(audio, n_windows=INDEX_WINDOWS_PER_SONG)
            batch = torch.stack([to_mel(w) for w in windows]).to(device)

            with torch.inference_mode():
                embs = model(batch).cpu().numpy().astype("float32")

            song_id = idx

            for window_idx, emb in enumerate(embs):
                all_embeddings.append(emb)
                all_song_ids.append(song_id)
                manifest.append(
                    {
                        "song_id": song_id,
                        "s3_key": key,
                        "window_idx": window_idx,
                    }
                )

            processed += 1

            if idx % 25 == 0 or idx == len(keys):
                logger.info(
                    "Embedding build progress processed=%s/%s skipped=%s vectors=%s",
                    idx,
                    len(keys),
                    skipped,
                    len(all_embeddings),
                )

        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.warning("Embedding build skipped key=%s error=%s", key, exc)
            continue

        finally:
            if path is not None:
                try:
                    os.remove(path)
                except OSError:
                    pass

    if not all_embeddings:
        raise ValueError("Could not build embeddings: all files failed")

    embeddings = np.asarray(all_embeddings, dtype="float32")
    song_ids = np.asarray(all_song_ids, dtype=np.int64)

    Path(EMBEDDINGS_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(SONG_IDS_PATH, song_ids)

    with open(SONG_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - started_total
    logger.info(
        "Embedding build finished processed=%s skipped=%s vectors=%s elapsed_sec=%.2f embeddings_path=%s",
        processed,
        skipped,
        len(all_embeddings),
        elapsed,
        EMBEDDINGS_PATH,
    )