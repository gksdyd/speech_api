import os
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import Image

MYSQL_MAIN_USERNAME = os.getenv("MYSQL_MAIN_USERNAME_TEST")
MYSQL_MAIN_PASSWORD = os.getenv("MYSQL_MAIN_PASSWORD_TEST")

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
    db_gen = get_db()
    db: Session = next(db_gen)
    result = 0

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
    except Exception as e:
        print(f"DB 오류: {e}")
        result = 1
    finally:
        db_gen.close()
    return result