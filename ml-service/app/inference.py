from collections import Counter

def aggregate_results(indices, song_ids):
    votes = []

    for idx_list in indices:
        for idx in idx_list:
            votes.append(song_ids[idx])

    if not votes:
        return None

    most_common = Counter(votes).most_common(1)[0]
    song_id, count = most_common

    return {
        "song_id": int(song_id),
        "confidence": count / len(votes)
    }
