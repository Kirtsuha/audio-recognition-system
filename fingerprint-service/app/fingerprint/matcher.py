from collections import defaultdict

from repository.models import Fingerprint


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def match(hashes, db, min_matches=20):
    votes = defaultdict(int)

    hash_values = [int(h) for h, _ in hashes]
    time_dict = {int(h): t for h, t in hashes}

    results = []
    for chunk in chunks(hash_values, 500):
        results += db.query(Fingerprint) \
            .filter(Fingerprint.hash.in_(chunk)) \
            .all()

    for r in results:
        delta = r.time_offset - time_dict[r.hash]
        votes[(r.track_id, delta)] += 1

    if not votes:
        return None

    best_match = max(votes.items(), key=lambda x: x[1])
    (track_id, delta), count = best_match

    if count < min_matches:
        return None

    return {
        'track_id': track_id,
        'offset': delta,
        'matches': count
    }