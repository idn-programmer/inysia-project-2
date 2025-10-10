from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from ..config import get_settings


class Base(DeclarativeBase):
    pass


def create_engine_and_session():
    settings = get_settings()
    db_url = settings.database_url
    if db_url.startswith("sqlite"):
        engine = create_engine(db_url, pool_pre_ping=True, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(db_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal


engine, SessionLocal = create_engine_and_session()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


