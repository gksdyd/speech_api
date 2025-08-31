import os

from datetime import datetime
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import *

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

def insert(path: str, file: UploadFile, uuid: str, size: int, lnrd_seq:str, type: int, ifmm_seq: str, sort: int, db: Session):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    db_content = LangRecodeUploaded(
        path=path,
        originalName=file.filename,
        uuidName=uuid,
        ext=uuid.split(".")[-1],
        size=size,
        pseq=lnrd_seq,
        sort=sort,
        type=type,
        delNy=0,
        regIp="1",
        regSeq=ifmm_seq,
        regDeviceCd=0,
        regDateTime=date,
        regDateTimeSvr=date,
    )

    db.add(db_content)

def study_usr_inst(lnrd_status_cd: int, lnrd_type_cd: int, lnrd_title: str , ifmm_seq: str, lnrd_run_time: int, db: Session):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    db_content = LangRecord(
        lnrdStatusCd=lnrd_status_cd,
        lnrdTypeCd=lnrd_type_cd,
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

def study_usr_updt(lnrd_status_cd: int, lnrdSeq: str, ifmm_seq: str, db: Session):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    record = db.query(LangRecord).filter(LangRecord.lnrdSeq == lnrdSeq).first()
    if not record:
        raise ValueError(f"Record with id {lnrdSeq} not found")

    record.lnrdStatusCd = lnrd_status_cd
    record.modIp = "0"
    record.modSeq = ifmm_seq
    record.modDeviceCd = 0
    record.modDateTime = date
    record.modDateTimeSvr = date

    return record

def script_usr_inst(contents: str, eng_contents: str, speaker: int , lnrd_seq: str, ifmm_seq: str, speaker_gender: int, db: Session):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    db_content = LangScript(
        lnscSpeakerGenderCd=speaker_gender,
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

def study_result_inst(contents: str, score: float, lnst_seq: str, lnsc_seq: str, ifmm_seq: str, db: Session):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    db_content = LangStudyResult(
        lnsrContents=contents,
        lnsrScore=score,
        lnsrDelNy=0,
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
        lnstSeq=lnst_seq,
        lnscSeq=lnsc_seq,
    )

    db.add(db_content)
    db.flush()

    return db_content.lnscSeq

def save_db_process(path: str, file: UploadFile, uuid: str, size: int, ifmm_seq: str, result_seperate: list, foreign_key: str):
    db_gen = get_db()
    db: Session = next(db_gen)

    try:
        study_usr_updt(1002, foreign_key, ifmm_seq, db)
        insert(path, file, uuid, size, foreign_key, 10, ifmm_seq, 1, db)
        for contents in result_seperate:
            script_usr_inst(contents[1], contents[2], contents[0], foreign_key, ifmm_seq, contents[3], db)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"DB 오류: {e}")
    finally:
        db.close()

def insert_db_lnrd_recoding(lnrd_status_cd: int, lnrd_type_cd: int, lnrd_title: str , ifmm_seq: str, lnrd_run_time: int):
    db_gen = get_db()
    db: Session = next(db_gen)

    try:
        foreign_key = study_usr_inst(lnrd_status_cd, lnrd_type_cd, lnrd_title, ifmm_seq, lnrd_run_time, db)
        db.commit()
        return foreign_key
    except Exception as e:
        db.rollback()
        print(f"DB 오류: {e}")
    finally:
        db.close()

def update_db_lnrd_recoding_for_empty_contents(ifmm_seq: str, foreign_key: str):
    db_gen = get_db()
    db: Session = next(db_gen)

    try:
        study_usr_updt(1003, foreign_key, ifmm_seq, db)
        db.commit()
        return foreign_key
    except Exception as e:
        db.rollback()
        print(f"DB 오류: {e}")
    finally:
        db.close()

def insert_db_study_result(path: str, file: UploadFile, uuid: str, size: int, contents: str, score: float, lnst_seq: str, lnsc_seq: str, ifmm_seq: str, sort: int):
    db_gen = get_db()
    db: Session = next(db_gen)

    try:
        foreign_key = study_result_inst(contents, score, lnst_seq, lnsc_seq, ifmm_seq, db)
        insert(path, file, uuid, size, foreign_key, 20, ifmm_seq, sort, db)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"DB 오류: {e}")
    finally:
        db.close()