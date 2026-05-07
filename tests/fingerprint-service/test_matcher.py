from types import SimpleNamespace

import fingerprint.matcher as matcher


class FakeColumn:
    def in_(self, values):
        return values


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, _condition):
        return self

    def all(self):
        return self.rows


class FakeDb:
    def __init__(self, rows):
        self.rows = rows

    def query(self, *_columns):
        return FakeQuery(self.rows)


def setup_module(_module):
    matcher.Fingerprint.hash = FakeColumn()


def test_retrieve_candidates_no_hashes():
    result = matcher.retrieve_candidates([], FakeDb([]))

    assert result["candidates"] == []
    assert result["reason"] == "no_hashes"


def test_retrieve_candidates_aggregates_offsets(monkeypatch):
    monkeypatch.setattr(matcher, "OFFSET_BIN", 1)
    monkeypatch.setattr(matcher, "OFFSET_TOLERANCE_FRAMES", 1)

    db = FakeDb(
        [
            (111, 10, 5),
            (111, 10, 6),
            (222, 20, 2),
        ]
    )
    result = matcher.retrieve_candidates([(111, 1), (222, 1)], db, top_k=2)

    assert result["query_hashes"] == 2
    assert result["unique_query_hashes"] == 2
    assert result["candidates"][0]["track_id"] == 10
    assert result["candidates"][0]["aligned_matches"] == 2
    assert result["best_aligned_matches"] == 2


def test_match_rejects_low_confidence(monkeypatch):
    monkeypatch.setattr(matcher, "retrieve_candidates", lambda hashes, db, top_k: {
        "query_hashes": 100,
        "unique_query_hashes": 80,
        "second_aligned_matches": 9,
        "candidates": [
            {
                "track_id": 1,
                "best_offset": 2,
                "best_offset_sec": 0.5,
                "aligned_matches": 3,
                "total_matches": 3,
                "coverage": 0.03,
                "confidence": 0.2,
            }
        ],
    })
    monkeypatch.setattr(matcher, "MIN_ALIGNED_MATCHES", 10)

    result = matcher.match([(1, 1)], object())

    assert result["matched"] is False
    assert result["reason"] == "low_confidence"


def test_match_accepts_strong_candidate(monkeypatch):
    monkeypatch.setattr(matcher, "MIN_ALIGNED_MATCHES", 2)
    monkeypatch.setattr(matcher, "MIN_QUERY_COVERAGE", 0.01)
    monkeypatch.setattr(matcher, "MIN_SCORE_GAP", 1.5)
    monkeypatch.setattr(matcher, "retrieve_candidates", lambda hashes, db, top_k: {
        "query_hashes": 100,
        "unique_query_hashes": 80,
        "second_aligned_matches": 2,
        "candidates": [
            {
                "track_id": 7,
                "best_offset": 3,
                "best_offset_sec": 1.0,
                "aligned_matches": 20,
                "total_matches": 24,
                "coverage": 0.2,
                "confidence": 0.9,
            }
        ],
    })

    result = matcher.match([(1, 1)], object())

    assert result["matched"] is True
    assert result["track_id"] == 7
    assert result["reason"] is None


def test_offset_bin_to_seconds(monkeypatch):
    monkeypatch.setattr(matcher, "OFFSET_BIN", 2)
    monkeypatch.setattr(matcher, "HOP_LENGTH", 512)
    monkeypatch.setattr(matcher, "SAMPLE_RATE", 16000)

    assert matcher.offset_bin_to_seconds(10) == 0.64
