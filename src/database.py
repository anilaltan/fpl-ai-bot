from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.config import settings

# SQLAlchemy Engine Oluştur (PostgreSQL)
engine = create_engine(settings.database_url)

# Session Sınıfı
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base Model Sınıfı
Base = declarative_base()

# Dependency (FastAPI için DB Bağlantısı Sağlayıcı)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
