import os
from fastapi import Depends, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from models import Image

MYSQL_MAIN_USERNAME = os.getenv("MYSQL_MAIN_USERNAME")
MYSQL_MAIN_PASSWORD = os.getenv("MYSQL_MAIN_PASSWORD")

DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_MAIN_USERNAME}:{MYSQL_MAIN_PASSWORD}"
    "@peter.czsiy02maq2z.ap-northeast-2.rds.amazonaws.com:3306/peterdb"
)

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def  insert(path: str, file: UploadFile, uuid: str, size: int):
    now = datetime.now()

    db_gen = get_db()
    db: Session = next(db_gen)

    try:
        db_content = Image(
            path=path,
            originalName=file.filename,
            uuidName=uuid,
            ext=uuid.split(".")[-1],
            size=size,
            pseq=1,
            sort=1,
            type=10,
            delNy=0,
        )

        db.add(db_content)
        db.commit()
        db.refresh(db_content)
    finally:
        db_gen.close()