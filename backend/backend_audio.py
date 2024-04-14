'''
Author: VideoPal Team
Date: 2024-03-06 18:58:50
LastEditors: VideoPal Team
LastEditTime: 2024-03-24 13:43:31
FilePath: /chengruilai/projects/VideoPal/backend/backend_audio.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import os
import time
import tqdm
from typing import List

from utils.database import db, AudioResult
from models.whisper_model import WhisperModel

parent_dir = os.path.dirname(__file__)


class AudioBackend:
    def __init__(self, whisper_model: WhisperModel, args) -> None:
        '''
        Audio backend for VideoPal.
        whispper_model: Whisper model
        args: Arguments
        '''
        self.whisper_model = whisper_model
        self.args = args
        
    def get_audio_info(self, filepath: str, video_info: List, n_sentences: int = 1):
        '''
        filepath: video file path e.g. ./data/test.mp4
        video_info: [vid, video_ename, video_length, filepath]
        n_sentences: number of sentences to be combined
        '''
        # vid, filename, video_length = get_video_info(filepath, db=db)
        start_time = time.perf_counter()
        print(f'\033[1;33mStarting Extract ASR Information of Vid: {video_info[0]}, Video: {video_info[1]}\033[0m')
        audio_result = self.whisper_model.model.transcribe(filepath, task = "translate")
        result = []
        tmp_result = []
        count = 0
        
        for segment in tqdm.tqdm(audio_result['segments']):
            if segment['no_speech_prob'] < 0.5:  
                tmp_result.append(segment)
                count += 1
                if not count % n_sentences:
                    result.append(AudioResult(vid = video_info[0],
                                              video_name = video_info[1],
                                              start_time = tmp_result[0]['start'], 
                                              end_time = tmp_result[-1]['end'], 
                                              content = ' '.join([segment['text'] for segment in tmp_result])))
                    tmp_result.pop(0)
        
        db.add_all(result)
        db.commit()
        print(f'\033[1;33mFinished After {time.perf_counter() - start_time} Seconds\033[0m')