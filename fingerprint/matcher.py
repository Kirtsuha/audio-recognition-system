from collections import defaultdict

from repository.models import Fingerprint


def match(hashes, db, min_matches=20):

    votes = defaultdict(int)

    for h, t_query in hashes:
        matches = db.query(Fingerprint).filter(Fingerprint.hash == h).all()
        for m in matches:
            delta = m.time_offset - t_query
            votes[(m.track_id, delta)] += 1

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