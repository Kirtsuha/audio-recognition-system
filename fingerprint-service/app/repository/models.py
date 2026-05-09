from sqlalchemy import Column, Integer, Text, ForeignKey, BigInteger, UniqueConstraint

from repository.database_config import Base
from sqlalchemy import Index

class Track(Base):
    __tablename__ = "track"

    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    artist = Column(Text, nullable=False)
    album = Column(Text, nullable=True)
    s3_key = Column(Text, nullable=False)
    duration_sec = Column(Integer, nullable=True)
    fingerprint_count = Column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("s3_key", name="track_s3_key_unique_idx"),
    )

class Fingerprint(Base):
    __tablename__ = "fingerprint"

    hash = Column(BigInteger, nullable=False, primary_key=True)
    track_id = Column(Integer, ForeignKey("track.id"), nullable=False, primary_key=True)
    time_offset = Column(Integer, nullable=False, primary_key=True)


    __table_args__ = (
        UniqueConstraint("hash", "track_id", "time_offset", name="fingerprint_unique_idx"),
        Index("fingerprint_hash_idx", "hash"),
    )
