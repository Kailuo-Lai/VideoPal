'''
Author: VideoPal Team
Date: 2024-03-06 19:01:21
LastEditors: VideoPal Team
LastEditTime: 2024-04-13 18:10:33
FilePath: /root/autodl-fs/projects/VideoPal/utils/database.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

from sqlalchemy import create_engine, Column, String, Text, DateTime, func, Integer, Float, LargeBinary, ForeignKey, \
    UniqueConstraint, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus

pymysql.install_as_MySQLdb()
Base = declarative_base()

def create_db_engine(echo):
    # Change to you database and password
    password = quote_plus('asd123456') 
    return create_engine(f'mysql+mysqldb://root:{password}@localhost/video_pal', echo=echo)


def create_db_session():
    engine = create_db_engine(False)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def create_db_and_tables():
    engine = create_db_engine(True)
    Base.metadata.create_all(engine, checkfirst=True)
    return True

class VideoInfo(Base):
    __tablename__ = 'video_info'
    vid = Column(Integer, primary_key=True, autoincrement=True)
    video_name = Column(VARCHAR(256), unique=True)
    video_length = Column(Float)
    filepath = Column(Text)
    
class VisualResult(Base):
    __tablename__ = 'visual_result'
    id = Column(Integer, primary_key=True, autoincrement=True)
    vid = Column(Integer, ForeignKey('video_info.vid'))
    video_name = Column(VARCHAR(256), ForeignKey('video_info.video_name'))
    time = Column(Float)
    clip_id = Column(Integer)
    tag = Column(Text)
    caption = Column(Text)
    OCR = Column(Text)
    
class AudioResult(Base):
    __tablename__ = 'audio_result'
    id = Column(Integer, primary_key=True, autoincrement=True)
    vid = Column(Integer, ForeignKey('video_info.vid'))
    video_name = Column(VARCHAR(256), ForeignKey('video_info.video_name'))
    start_time = Column(Float)
    end_time = Column(Float)
    content = Column(Text)
    
class VideoCapResult(Base):
    __tablename__ = 'video_cap_result'
    id = Column(Integer, primary_key=True, autoincrement=True)
    vid = Column(Integer, ForeignKey('video_info.vid'))
    video_name = Column(VARCHAR(256), ForeignKey('video_info.video_name'))
    start_time = Column(Float)
    end_time = Column(Float)
    content = Column(Text)

class VideoSummaryResult(Base):
    __tablename__ = 'video_summary_result'
    id = Column(Integer, primary_key=True, autoincrement=True)
    vid = Column(Integer, ForeignKey('video_info.vid'))
    video_name = Column(VARCHAR(256), ForeignKey('video_info.video_name'))
    content = Column(Text)
    
class VideoEventsSummaryResult(Base):
    __tablename__ = 'video_events_summary_result'
    id = Column(Integer, primary_key=True, autoincrement=True)
    vid = Column(Integer, ForeignKey('video_info.vid'))
    video_name = Column(VARCHAR(256), ForeignKey('video_info.video_name'))
    start_time = Column(Float)
    end_time = Column(Float)
    content = Column(Text)
    
class VideoResultComplete(Base):
    __tablename__ = 'video_complete_result'
    id = Column(Integer, primary_key=True, autoincrement=True)
    vid = Column(Integer, ForeignKey('video_info.vid'))
    video_name = Column(VARCHAR(256), ForeignKey('video_info.video_name'))
    video_length = Column(Float)
    filepath = Column(Text)
    

db = create_db_session()
create_db_and_tables()

