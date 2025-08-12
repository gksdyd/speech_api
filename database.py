import os

from datetime import datetime
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import LangRecodeUploaded , LangRecoding , LangScript

MYSQL_MAIN_USERNAME = os.getenv("MYSQL_MAIN_USERNAME")
MYSQL_MAIN_PASSWORD = os.getenv("MYSQL_MAIN_PASSWORD")

DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_MAIN_USERNAME}:{MYSQL_MAIN_PASSWORD}"
    "@kryx-tt.cpyq48w4gz7g.ap-northeast-2.rds.amazonaws.com:33067/grace"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=20,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def insert(path: str, file: UploadFile, uuid: str, size: int, lnrd_seq:str, ifmm_seq, db: Session):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    db_content = LangRecodeUploaded(
        path=path,
        originalName=file.filename,
        uuidName=uuid,
        ext=uuid.split(".")[-1],
        size=size,
        pseq=lnrd_seq,
        sort=1,
        type=10,
        delNy=0,
        regIp="1",
        regSeq=ifmm_seq,
        regDeviceCd=0,
        regDateTime=date,
        regDateTimeSvr=date,
    )

    db.add(db_content)

def study_usr_inst(lnrd_status_cd: int, lnrd_type_ct: int, lnrd_title: str , ifmm_seq: str, lnrd_run_time: int, db: Session):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    db_content = LangRecoding(
        lnrdStatusCd=lnrd_status_cd,
        lnrdTypeCt=lnrd_type_ct,
        lnrdTitle=lnrd_title,
        lnrdRunTime=lnrd_run_time,
        lnrdDelNy=0,
        regIp="1",
        regSeq=ifmm_seq,
        regDeviceCd=0,
        regDateTime=date,
        regDateTimeSvr=date,
        modIp="0",
        modSeq=ifmm_seq,
        modDeviceCd=0,
        modDateTime=date,
        modDateTimeSvr=date,
        ifmmSeq=ifmm_seq,
    )

    db.add(db_content)
    db.flush()

    return db_content.lnrdSeq

def script_usr_inst(contents: str, eng_contents: str, speaker: int , lnrd_seq: str, ifmm_seq, db: Session):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    db_content = LangScript(
        lnscContents=contents,
        lnscContentsEng=eng_contents,
        lnscSpeakerCd=speaker,
        lnscDelNy=0,
        regIp="1",
        regSeq=ifmm_seq,
        regDeviceCd=0,
        regDateTime=date,
        regDateTimeSvr=date,
        modIp="0",
        modSeq=ifmm_seq,
        modDeviceCd=0,
        modDateTime=date,
        modDateTimeSvr=date,
        lnrdSeq=lnrd_seq,
    )

    db.add(db_content)
    db.flush()

    return db_content.lnscSeq

def save_db_process(path: str, file: UploadFile, uuid: str, size: int, lnrd_status_cd: int, lnrd_type_ct: int, lnrd_title: str , ifmm_seq: str, result_seperate: list, lnrd_run_time: int):
    if len(result_seperate) == 0:
        print(f"음성 추출 실패!!")
        return

    db_gen = get_db()
    db: Session = next(db_gen)

    try:
        foreign_key = study_usr_inst(lnrd_status_cd, lnrd_type_ct, lnrd_title, ifmm_seq, lnrd_run_time, db)
        insert(path, file, uuid, size, foreign_key, ifmm_seq, db)
        for contents in result_seperate:
            script_usr_inst(contents[1], contents[2], contents[0], foreign_key, ifmm_seq, db)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"DB 오류: {e}")
    finally:
        db_gen.close()
