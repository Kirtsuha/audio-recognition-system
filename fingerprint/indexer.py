

def index_track(track_id, hashes, db):
    for h, t in hashes:
        if h not in db:
            db[h] = []
        db[h].append((track_id, t))