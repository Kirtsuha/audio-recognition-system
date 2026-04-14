\
from collections import Counter

def aggregate_results(indices, song_ids):

    votes = []

    for idx_list in indices:
        for idx in idx_list:
            votes.append(song_ids[idx])

    most_common = Counter(votes).most_common(1)

    return most_common[0][0]