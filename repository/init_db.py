from database_config import engine
from repository.models import Base

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)