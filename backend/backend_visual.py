'''
Author: VideoPal Team
Date: 2024-03-09 21:42:09
LastEditors: VideoPal Team
LastEditTime: 2024-04-14 12:31:37
FilePath: /root/autodl-fs/projects/VideoPal/backend/backend_visual.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import cv2
import torch
import time
import tqdm
import numpy as np
from typing import List
from PIL import Image
import os
import shutil
from paddleocr import PaddleOCR

from models.clip_model import CLIPModel
from models.tag2text_model import Tag2TextModel
from models.kts_model import KTSModel
from models.video_caption_model import VideoCaptionModel
from models.translate_model import Translator
from utils.database import db, VisualResult, VideoCapResult
from utils.utils  import cut_video_by_ffmpeg
from utils.inputparser import translate_zh_en

class VisualBackend:
    def __init__(self, 
                 clip_model: CLIPModel, 
                 tag2text_model: Tag2TextModel, 
                 kts_model: KTSModel,
                 ocr: PaddleOCR,
                 video_caption_model : VideoCaptionModel,
                 translator: Translator,
                 args):
        '''
        Visual backend for VideoPal.
        clip_model: CLIP model
        tag2text_model: Tag2Text model
        kts_model: KTS model
        ocr: OCR model
        Video_Caption_Model: timesformer-gpt2-video-captioning
        translator: Translate model
        args: Arguments
            inter_seconds: The smallest time gap between successive clips, in seconds.
        '''
        self.clip_model = clip_model
        self.tag2text_model = tag2text_model
        self.kts_model = kts_model
        self.args = args
        self.inter_seconds = args.inter_seconds
        self.video_caption_model = video_caption_model
        self.translator = translator
        self.ocr = ocr
    
    def get_static_info_of_whole_video(self, filepath: str, video_info: List):
        '''
        Extract static information of one video per {inter_seconds} seconds.
        filepath: video file path e.g. ./data/test.mp4
        video_info: [vid, video_name, video_length, filepath]
        '''
        try: 
            os.mkdir(f'./tmp/{video_info[0]}_{video_info[1]}')
        except:
            shutil.rmtree(f'./tmp/{video_info[0]}_{video_info[1]}')  
            os.mkdir(f'./tmp/{video_info[0]}_{video_info[1]}')
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = int(fps) * self.inter_seconds

        clip_features = []
        result = []
        count = 0
        start_time = time.perf_counter()
        print(f'\033[1;33mStarting Extract Static Information from Images of Vid: {video_info[0]}, Video: {video_info[1]}\033[0m')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            if current_frame_pos % int(fps) == 0 or current_frame_pos == 1 or current_frame_pos == frame_count - 1:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # CLIP part
                inputs = self.clip_model.processor(images=image, return_tensors="pt").pixel_values
                with torch.inference_mode():
                    feat = self.clip_model.vision_model(inputs)['image_embeds']
                    clip_features.append(feat.cpu().numpy())
                count += 1
            
            # Get first frame and last frame
            if current_frame_pos % sample_rate == 0 or current_frame_pos == 1 or current_frame_pos == frame_count - 1:
                # Tag2Text part
                inputs = self.tag2text_model.transform(Image.fromarray(image)).unsqueeze(0)
                with torch.inference_mode():
                    caption, tag_predict = self.tag2text_model.model.generate(inputs,
                                                                              tag_input=None,
                                                                              max_length=50,
                                                                              return_tag_predict=True)
                    
                # OCR part
                ocr_info = []
                ocr_result = self.ocr.ocr(image, cls=True)
                for res in ocr_result:
                    if res is not None:
                        for line in res:
                            try:
                                if line[1][0]:  # Change to 'if line[1][0]' which can automatically handle None and empty strings
                                    ocr_info.append(line[1][0])
                            except IndexError:
                                # Errors can be logged or passed
                                pass  
                ocr_info = ','.join(ocr_info)
                ocr_info = translate_zh_en(ocr_info, self.translator)
                # Save to database
                result.append(VisualResult(vid = video_info[0],
                                           video_name = video_info[1],
                                           time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps,
                                           clip_id = count - 1,
                                           tag = tag_predict[0].replace(' | ', ','),
                                           caption = caption[0],
                                           OCR = ocr_info))
                    
            
                

        clip_features = np.concatenate(clip_features, axis=0) 
        np.savez_compressed(f"./tmp/{video_info[0]}_{video_info[1]}/clip_embed.npz", features=clip_features)
        
        db.add_all(result)
        db.commit()
        
        print(f'\033[1;33mFinished After {time.perf_counter() - start_time} Seconds\033[0m')
        
    def get_static_info_of_one_frame(self, filepath: str, second: int):
        '''
        Extract static information of one frame.
        filepath: video file path e.g. ./data/test.mp4
        second: second of the frame to be extracted
        '''
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Tag2Text part
            inputs = self.tag2text_model.transform(Image.fromarray(image)).unsqueeze(0)
            with torch.inference_mode():
                caption, tag_predict = self.tag2text_model.model.generate(inputs,
                                                                            tag_input=None,
                                                                            max_length=50,
                                                                            return_tag_predict=True)
            
            # OCR part
            ocr_info = []
            ocr_result = self.ocr.ocr(image, cls=True)
            for res in ocr_result:
                if res is not None:
                    for line in res:
                        try:
                            if line[1][0]:  # Change to 'if line[1][0]' which can automatically handle None and empty strings
                                ocr_info.append(line[1][0])
                        except IndexError:
                            # Errors can be logged or passed
                            pass
            ocr_info = ','.join(ocr_info)
            ocr_info = translate_zh_en(ocr_info, self.translator)
        
            return tag_predict[0].replace(' | ', ', '), caption[0], ocr_info
        
        else:
            return None, None, None
    
    def segment_video(self, video_info: List, inter_seconds: int = 1):
        '''
        segment video using KTS model.
        video_info: [vid, video_name, video_length, filepath]
        inter_seconds: The smallest time gap between successive clips, in seconds.
        '''
        video_features = np.load(f"./tmp/{video_info[0]}_{video_info[1]}/clip_embed.npz")['features']
        seg_windows = self.kts_model(video_features, video_info[2], inter_seconds)
        return seg_windows
    
    @staticmethod
    def segment_video_(video_info: List, kts_model: KTSModel, inter_seconds: int = 1):
        '''
        segment video using KTS model. (static method)
        video_info: [vid, video_ename, video_length, filepath]
        inter_seconds: The smallest time gap between successive clips, in seconds.
        '''
        video_features = np.load(f"./tmp/{video_info[0]}_{video_info[1]}/clip_embed.npz")['features']
        seg_windows = kts_model(video_features, video_info[2], inter_seconds)
        return seg_windows
        
    def get_dynamic_info_of_one_segment(self, filepath, video_info: List, start_time: float, end_time: float):
        '''
        get dynamic information of one segment.
        filepath: video file path e.g. ./data/test.mp4
        video_info: [vid, video_name, video_length, filepath]
        start_time: start time of the segment in seconds
        end_time: end time of the segment in seconds
        '''
        vid, video_name, video_length, filepath = video_info
        assert start_time <= video_length
        cut_start_time = int(start_time)
        cut_end_time = min(int(video_length), int(end_time))
        
        cut_saved_path = cut_video_by_ffmpeg(filepath, cut_start_time, cut_end_time, f'./tmp/{vid}_{video_name}/cut_store')
        caption = self.video_caption_model(cut_saved_path)
        shutil.rmtree( f'./tmp/{vid}_{video_name}/cut_store') 
        return caption

    def get_dynamic_info_of_whole_video(self, filepath: str, video_info: List, clip_length: int = 20):
        '''
        filepath: video file path e.g. ./data/test.mp4
        video_info: [vid, video_name, video_length, filepath]
        clip_length: length of per clip to translate to caption
        '''
        # vid, filename, video_length = get_video_info(filepath, db=db)
        print(f'\033[1;33mStarting Extract Video Caption Information of Vid: {video_info[0]}, Video: {video_info[1]}\033[0m')
        clip_length = self.args.inter_seconds_video_cap
        vid, video_name, video_length, filepath = video_info
        result = []
        ncuts = int(video_length // clip_length) + 1

        start_time = time.perf_counter()
        for part in tqdm.tqdm(range(ncuts)):
            start = clip_length * part
            end = min(clip_length * (part + 1), video_length)
            try:
                caption = self.get_dynamic_info_of_one_segment(filepath, video_info, start , end)
                result.append(VideoCapResult(vid = vid, video_name = video_name, start_time = int(start), end_time = int(end), content = caption))
            except:
                pass
        
        db.add_all(result)
        db.commit()
        print(f'\033[1;33mFinished After {time.perf_counter() - start_time} Seconds\033[0m')
            