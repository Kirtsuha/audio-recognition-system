from repository.models import Fingerprint


def index_track(track_id, hashes, db):
    for h, t in hashes:
        db.add(Fingerprint(
            hash=h,
            track_id=track_id,
            time_offset=t
        ))
    db.commit()