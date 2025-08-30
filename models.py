from sqlalchemy import Column, Integer, String, DateTime, SmallInteger, Double
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class LangRecodeUploaded(Base):
    __tablename__ = 'langRecordUploaded'  # MySQL 테이블 이름

    seq = Column(Integer, primary_key=True, autoincrement=True)  # PK, AI
    type = Column(Integer)
    defaultNy = Column(SmallInteger)  # tinyint
    sort = Column(Integer)
    path = Column(String(200))
    originalName = Column(String(200))
    uuidName = Column(String(200))
    ext = Column(String(45))
    size = Column(Integer)
    delNy = Column(SmallInteger)
    pseq = Column(Integer)
    regIp = Column(String(100))
    regSeq = Column(Integer)
    regDeviceCd = Column(Integer)
    regDateTime = Column(DateTime)
    regDateTimeSvr = Column(DateTime)

class LangRecord(Base):
    __tablename__ = 'langRecord'  # MySQL 테이블 이름

    lnrdSeq = Column(Integer, primary_key=True, autoincrement=True)  # PK, AI
    lnrdStatusCd = Column(Integer)
    lnrdTypeCd = Column(Integer)
    lnrdTitle = Column(String(45))
    lnrdRunTime = Column(Integer)
    lnrdDelNy = Column(SmallInteger)  # tinyint
    regIp = Column(String(100))
    regSeq = Column(Integer)
    regDeviceCd = Column(Integer)
    regDateTime = Column(DateTime)
    regDateTimeSvr = Column(DateTime)
    modIp = Column(String(100))
    modSeq = Column(Integer)
    modDeviceCd = Column(Integer)
    modDateTime = Column(DateTime)
    modDateTimeSvr = Column(DateTime)
    ifmmSeq = Column(String)

class LangScript(Base):
    __tablename__ = 'langScript'  # MySQL 테이블 이름

    lnscSeq = Column(Integer, primary_key=True, autoincrement=True)  # PK, AI
    lnscSpeakerGenderCd = Column(Integer)
    lnscContents = Column(String(2000))
    lnscContentsEng = Column(String(2000))
    lnscContentsLang1 = Column(String(2000))
    lnscContentsLang2 = Column(String(2000))
    lnscSpeakerCd = Column(Integer)
    lnscDesc = Column(String(2000))
    lnscDelNy = Column(SmallInteger)  # tinyint
    regIp = Column(String(100))
    regSeq = Column(Integer)
    regDeviceCd = Column(Integer)
    regDateTime = Column(DateTime)
    regDateTimeSvr = Column(DateTime)
    modIp = Column(String(100))
    modSeq = Column(Integer)
    modDeviceCd = Column(Integer)
    modDateTime = Column(DateTime)
    modDateTimeSvr = Column(DateTime)
    lnrdSeq = Column(Integer)

class LangStudyResult(Base):
    __tablename__ = 'langStudyResult'  # MySQL 테이블 이름

    lnsrSeq = Column(Integer, primary_key=True, autoincrement=True)  # PK, AI
    lnsrContents = Column(String(2000))
    lnsrScore = Column(Double)
    lnsrDesc = Column(String(2000))
    lnsrDelNy = Column(SmallInteger)    # tinyint
    regIp = Column(String(100))
    regSeq = Column(Integer)
    regDeviceCd = Column(Integer)
    regDateTime = Column(DateTime)
    regDateTimeSvr = Column(DateTime)
    modIp = Column(String(100))
    modSeq = Column(Integer)
    modDeviceCd = Column(Integer)
    modDateTime = Column(DateTime)
    modDateTimeSvr = Column(DateTime)
    lnstSeq = Column(Integer)
    lnscSeq = Column(Integer)