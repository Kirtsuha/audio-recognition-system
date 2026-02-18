from sqlalchemy import Column, Integer, Text, ForeignKey, BigInteger, PrimaryKeyConstraint

from repository.database_config import Base


class Track(Base):
    __tablename__ = "track"

    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    artist = Column(Text, nullable=False)

class Fingerprint(Base):
    __tablename__ = "fingerprint"

    hash = Column(BigInteger, primary_key=True)
    track_id = Column(Integer, ForeignKey("track.id"), nullable=False)
    time_offset = Column(Integer, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("hash", "track_id", "time_offset"),
    )