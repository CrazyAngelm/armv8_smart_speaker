import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, declarative_base

engine = db.create_engine('sqlite:///backend/database.db', echo=True)

Base = declarative_base()

def init_db():
    Base.metadata.create_all(engine)

# Создаем сессию
SessionLocal = sessionmaker(bind=engine)

session = SessionLocal()
