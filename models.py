from sqlalchemy import Column, Integer, String, DateTime, SmallInteger
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class langRecodeUploaded(Base):
    __tablename__ = 'langRecodeUploaded'  # MySQL 테이블 이름

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

class langRecoding(Base):
    __tablename__ = 'langRecoding'  # MySQL 테이블 이름

    lnrdSeq = Column(Integer, primary_key=True, autoincrement=True)  # PK, AI
    lnrdStatusCd = Column(Integer)
    lnrdTypeCt = Column(Integer)
    lnrdTitle = Column(String(45))
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