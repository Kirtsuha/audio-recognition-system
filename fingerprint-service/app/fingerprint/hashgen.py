from collections import defaultdict

from config.config import FAN_VALUE, MIN_DELTA_T, MAX_DELTA_T


def pack_hash(f1: int, f2: int, delta_t: int) -> int:
    return ((int(f1) & 0xFFF) << 20) | ((int(f2) & 0xFFF) << 8) | (int(delta_t) & 0xFF)


def generate_hashes(peaks):
    peaks = sorted(peaks, key=lambda x: (x[0], x[1]))

    by_time = defaultdict(list)
    for t, f in peaks:
        by_time[int(t)].append(int(f))

    times = sorted(by_time.keys())
    hashes = []

    for t1 in times:
        anchors = by_time[t1]

        future = []
        for t2 in times:
            dt = t2 - t1
            if dt < MIN_DELTA_T:
                continue
            if dt > MAX_DELTA_T:
                break

            for f2 in by_time[t2]:
                future.append((dt, f2))

        future = future[:FAN_VALUE]

        for f1 in anchors:
            for dt, f2 in future:
                hashes.append((pack_hash(f1, f2, dt), t1))

    return hashes