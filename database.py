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
    echo=True,
    # pool_pre_ping=True
)

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
        # db.execute("SET SESSION max_allowed_packet = 67108864")

        db_content = LangRecodeUploaded(
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
        result = -1
    finally:
        db_gen.close()

    return result

async def  study_usr_inst(lnrd_status_cd: int, lnrd_type_ct: int, lnrd_title: str , ifmm_seq: str):
    db_gen = get_db()
    db: Session = next(db_gen)
    result = 0
    lnrd_seq = None

    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:

        db_content = LangRecoding(
            lnrdStatusCd=lnrd_status_cd,
            lnrdTypeCt=lnrd_type_ct,
            lnrdTitle=lnrd_title,
            lnrdDelNy=0,
            regIp="1",
            regSeq=10,
            regDeviceCd=0,
            regDateTime=date,
            regDateTimeSvr=date,
            modIp="0",
            modSeq=0,
            modDeviceCd=0,
            modDateTime=date,
            modDateTimeSvr=date,
            ifmmSeq=ifmm_seq,
        )

        db.add(db_content)
        db.commit()
        db.refresh(db_content)
        lnrd_seq = db_content.lnrdSeq
    except Exception as e:
        print(f"DB 오류: {e}")
        result = -1
    finally:
        db_gen.close()

    return lnrd_seq if lnrd_seq else result

async def  script_usr_inst(contents: str, eng_contents: str, speaker: int , lnrd_seq: str):
    db_gen = get_db()
    db: Session = next(db_gen)
    result = 0
    lnsc_seq = None

    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        db_content = LangScript(
            lnscContents=contents,
            lnscContentsEng=eng_contents,
            lnscSpeakerCd=speaker,
            lnscDelNy=0,
            regIp="1",
            regSeq=10,
            regDeviceCd=0,
            regDateTime=date,
            regDateTimeSvr=date,
            modIp="0",
            modSeq=0,
            modDeviceCd=0,
            modDateTime=date,
            modDateTimeSvr=date,
            lnrdSeq=lnrd_seq,
        )

        db.add(db_content)
        db.commit()
        db.refresh(db_content)
        lnsc_seq = db_content.lnscSeq
    except Exception as e:
        print(f"DB 오류: {e}")
        result = -1
    finally:
        db_gen.close()

    return lnsc_seq if lnsc_seq else result