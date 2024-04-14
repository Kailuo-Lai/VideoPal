'''
Author: VideoPal Team
Date: 2024-03-02 11:22:38
LastEditors: VideoPal Team
LastEditTime: 2024-03-24 23:05:00
FilePath: /chengruilai/projects/VideoPal/utils/utils.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import signal
import os
import contextlib
import subprocess

from sqlalchemy.sql import func
from sqlalchemy.orm import Session

from utils.database import VideoInfo



def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
        
@contextlib.contextmanager
def new_cd(x):
    d = os.getcwd()
    # This could raise an exception, but it's probably
    # best to let it propagate and let the caller
    # deal with it, since they requested x
    os.chdir(x)

    try:
        yield

    finally:
        # This could also raise an exception, but you *really*
        # aren't equipped to figure out what went wrong if the
        # old working directory can't be restored.
        os.chdir(d)
        
def get_video_length(filepath):
    output = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath], stdout=subprocess.PIPE)
    video_length = float(output.stdout)
    return video_length

def get_video_info(filepath: str, db: Session):
    filename = filepath.split('/')[-1]
    video_length = get_video_length(filepath)
    print(f'\033[1;33mVideo Name: {filename}, Video Length: {video_length}...\033[0m')
    db.add(VideoInfo(video_name = filename, video_length = video_length, filepath = filepath))
    db.commit()
    vid = db.query(func.max(VideoInfo.vid)).filter(VideoInfo.video_name == filename).first()[0]
    return vid, filename, video_length, filepath

def cut_video_by_ffmpeg(video_path, start_time, end_time, save_dir, save_format = 'mp4'):
    os.makedirs(save_dir, exist_ok=True)
    
    assert start_time < end_time
    write_to_path = f'{save_dir}/{start_time}-{end_time}.{save_format}'
    subprocess.run(['ffmpeg', '-ss', str(start_time), '-i',video_path,'-t', str(end_time-start_time), 
                    '-c:v','copy', '-c:a', 'copy', '-y', write_to_path], capture_output=True)
    return write_to_path