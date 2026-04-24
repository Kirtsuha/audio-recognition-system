import json
import logging
from pathlib import Path

import numpy as np
import torch

from app.model import AudioEncoder
from pipeline.config import INDEX_WINDOWS_PER_SONG
from pipeline.dataset import load_prepared_manifest, extract_uniform_index_windows
from pipeline.experiment_utils import set_experiment_seed
from pipeline.to_mel import to_mel

logger = logging.getLogger("ml-pipeline.embed-build")


def build_embeddings_from_prepared(
    model_path: str | Path,
    embeddings_out: str | Path,
    song_ids_out: str | Path,
    manifest_out: str | Path,
    train_limit: int = 0,
    val_limit: int = 0,
    test_limit: int = 0,
    index_windows_override: int | None = None,
) -> None:
    set_experiment_seed()

    items = load_prepared_manifest(
        train_limit=train_limit,
        val_limit=val_limit,
        test_limit=test_limit,
    )
    if not items:
        raise ValueError("Prepared manifest is empty")

    windows_per_song = index_windows_override if index_windows_override is not None else INDEX_WINDOWS_PER_SONG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioEncoder().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_embeddings = []
    all_song_ids = []
    out_manifest = []

    for idx, row in enumerate(items, start=1):
        audio = np.load(row["prepared_path"]).astype("float32")
        windows = extract_uniform_index_windows(audio, n_windows=windows_per_song)
        batch = torch.stack([to_mel(w) for w in windows]).to(device)

        with torch.inference_mode():
            embs = model(batch).cpu().numpy().astype("float32")

        for window_idx, emb in enumerate(embs):
            all_embeddings.append(emb)
            all_song_ids.append(int(row["track_id"]))
            out_manifest.append(
                {
                    "track_id": int(row["track_id"]),
                    "s3_key": row["s3_key"],
                    "prepared_path": row["prepared_path"],
                    "split": row["split"],
                    "window_idx": window_idx,
                }
            )

        if idx % 250 == 0 or idx == len(items):
            logger.info(
                "Embedding build progress processed=%s/%s vectors=%s",
                idx,
                len(items),
                len(all_embeddings),
            )

    embeddings = np.asarray(all_embeddings, dtype="float32")
    song_ids = np.asarray(all_song_ids, dtype=np.int64)

    embeddings_out = Path(embeddings_out)
    song_ids_out = Path(song_ids_out)
    manifest_out = Path(manifest_out)

    embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_out, embeddings)
    np.save(song_ids_out, song_ids)

    with manifest_out.open("w", encoding="utf-8") as f:
        json.dump(out_manifest, f, ensure_ascii=False, indent=2)

    logger.info(
        "Embedding build finished tracks=%s vectors=%s embeddings=%s song_ids=%s index_windows_per_song=%s",
        len(items),
        len(all_embeddings),
        embeddings_out,
        song_ids_out,
        windows_per_song,
    )