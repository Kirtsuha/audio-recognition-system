import hashlib

from config.config import FAN_VALUE

def hash_triplet(f1, f2, delta_t):
    s = f"{f1}|{f2}|{delta_t}"
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def generate_hashes(peaks):
    peaks = sorted(peaks, key=lambda x: x[0])
    hashes = []

    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        for j in range(1, FAN_VALUE):
            if i + j < len(peaks):
                t2, f2 = peaks[i + j]
                delta_t = t2 - t1
                h = hash_triplet(f1, f2, delta_t)
                hashes.append((h, t1))
    return hashes
