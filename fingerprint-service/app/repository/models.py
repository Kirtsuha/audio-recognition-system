from sqlalchemy import Column, Integer, Text, ForeignKey, BigInteger, PrimaryKeyConstraint, UniqueConstraint

from repository.database_config import Base


class Track(Base):
    __tablename__ = "track"

    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    artist = Column(Text, nullable=False)
    s3_key = Column(Text, nullable=False)

class Fingerprint(Base):
    __tablename__ = "fingerprint"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hash = Column(BigInteger, nullable=False, index=True)
    track_id = Column(Integer, ForeignKey("track.id"), nullable=False)
    time_offset = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("hash", "track_id", "time_offset", name="fingerprint_unique_idx"),
    )