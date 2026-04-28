from collections import defaultdict

from config.config import (
    FAN_VALUE,
    MIN_DELTA_T,
    MAX_DELTA_T,
    FREQ_BIN_SIZE,
    DELTA_T_BIN_SIZE,
)


def quantize_freq(f: int) -> int:
    return int(f) // FREQ_BIN_SIZE


def quantize_delta_t(delta_t: int) -> int:
    return int(round(int(delta_t) / DELTA_T_BIN_SIZE))


def pack_hash(f1: int, f2: int, delta_t: int) -> int:
    qf1 = quantize_freq(f1)
    qf2 = quantize_freq(f2)
    qdt = quantize_delta_t(delta_t)

    return ((qf1 & 0xFFF) << 20) | ((qf2 & 0xFFF) << 8) | (qdt & 0xFF)


def generate_hashes(peaks):
    peaks = sorted(peaks, key=lambda x: (x[0], x[1]))

    by_time = defaultdict(list)
    for t, f in peaks:
        by_time[int(t)].append(int(f))

    times = sorted(by_time.keys())
    hashes = []

    for time_idx, t1 in enumerate(times):
        future = []

        for t2 in times[time_idx + 1:]:
            dt = t2 - t1

            if dt < MIN_DELTA_T:
                continue

            if dt > MAX_DELTA_T:
                break

            for f2 in by_time[t2]:
                future.append((dt, f2))

        if not future:
            continue

        if len(future) > FAN_VALUE:
            step = len(future) / FAN_VALUE
            future = [future[int(i * step)] for i in range(FAN_VALUE)]

        for f1 in by_time[t1]:
            for dt, f2 in future:
                hashes.append((pack_hash(f1, f2, dt), t1))

    return hashes