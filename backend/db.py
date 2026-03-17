import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

_raw_url = os.getenv("DATABASE_URL", "")

# SQLAlchemy requires postgresql+psycopg2:// scheme
# Also strip channel_binding param (not supported by psycopg2)
if _raw_url.startswith("postgresql://"):
    _url = _raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    _url = _raw_url

# Remove unsupported params
for _param in ("channel_binding=require", "channel_binding=prefer"):
    _url = _url.replace(f"&{_param}", "").replace(f"?{_param}&", "?").replace(f"?{_param}", "")

engine = create_engine(_url, pool_pre_ping=True) if _url else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()


def create_tables():
    if engine:
        Base.metadata.create_all(bind=engine)


def get_session():
    if not SessionLocal:
        return None
    return SessionLocal()
