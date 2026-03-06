from fingerprint.matcher import match
from repository.models import Fingerprint

class FakeDB:
    def __init__(self, records):
        self.records = records
    def query(self, model):
        return self
    def filter(self, cond):
        return self
    def all(self):
        return self.records

def test_match_no_votes():
    hashes = [(123, 0)]
    db = FakeDB([])
    result = match(hashes, db)
    assert result is None

def test_match_found():
    hashes = [(123, 0)]
    db = FakeDB([Fingerprint(hash=123, track_id=1, time_offset=0)])
    result = match(hashes, db, min_matches=1)
    assert result["track_id"] == 1
    assert result["matches"] == 1