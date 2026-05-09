import os
from typing import List

import psycopg


def get_tracks_database_url() -> str:
    value = os.getenv("TRACKS_DATABASE_URL")
    if not value:
        raise RuntimeError("TRACKS_DATABASE_URL is not set")
    return value


def load_tracks() -> List[dict]:
    sql = "SELECT id, s3_key FROM track"

    rows: List[dict] = []
    with psycopg.connect(get_tracks_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            for track_id, s3_key in cur.fetchall():
                if s3_key:
                    rows.append(
                        {
                            "track_id": int(track_id),
                            "s3_key": str(s3_key),
                        }
                    )
    return rows
