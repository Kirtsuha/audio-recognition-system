import hashlib

from config.config import FAN_VALUE, MAX_DELTA_T


def hash_triplet(f1, f2, delta_t):
    s = f"{f1}|{f2}|{delta_t}"
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

def pack_hash(f1: int, f2: int, delta_t: int) -> int:
    return (f1 << 23) | (f2 << 24) | (delta_t)


def generate_hashes(peaks):
    peaks = sorted(peaks, key=lambda x: x[0])
    hashes = []

    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        for j in range(1, FAN_VALUE):
            if i + j < len(peaks):
                t2, f2 = peaks[i + j]
                delta_t = t2 - t1
                if 0 < delta_t <= MAX_DELTA_T:
                    h = pack_hash(f1, f2, delta_t)
                    hashes.append((h, t1))
    print("peaks:", len(peaks))
    print("hashes:", len(hashes))
    return hashes
