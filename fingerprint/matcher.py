from collections import defaultdict


def match(hashes, db, min_matches=20):
    """
    :param hashes: List[(hash, time_offset)]
    :param db: dict[hash] -> List[(track_id, time_offset)]
    :param min_matches:
    :return:
    """

    votes = defaultdict(int)

    for h, t_query in hashes:
        if h not in db:
            continue

        for track_id, t_db in db[h]:
            delta = t_db - t_query
            votes[(track_id, delta)] += 1

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