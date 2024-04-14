'''
Author: VideoPal Team
Date: 2024-03-22 14:33:21
LastEditors: VideoPal Team
LastEditTime: 2024-04-11 22:22:33
FilePath: /root/autodl-fs/projects/VideoPal/models/video_caption_model.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import av
import os
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from ipex_llm.optimize import optimize_model
from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)

class VideoCaptionModel:
    def __init__(self, args) -> None:
        with new_cd(parent_dir):
            self.args = args
            self.image_processor = AutoImageProcessor.from_pretrained("../checkpoints/videomae-base")
            self.caption_model = VisionEncoderDecoderModel.from_pretrained("../checkpoints/timesformer-gpt2-video-captioning")
            if args.video_caption_low_bit:            
                self.caption_model = optimize_model(self.caption_model, low_bit='sym_int4')
            if torch.cuda.is_available():
                self.image_processor.cuda()
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.gen_kwargs = {
                            "min_length": 10, 
                            "max_length": 20, 
                            "num_beams": args.num_beams,
                        }
                    
    def __call__(self, video_path) -> str:        
        container = av.open(video_path)
        num_frames = container.streams.video[0].frames
        num_draw_frames = self.caption_model.config.encoder.num_frames 
        indices = set(np.linspace(0, num_frames, num=num_draw_frames, endpoint=False).astype(np.int64))
        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
                
        container.close()
        pixel_values = self.image_processor(frames, return_tensors="pt").pixel_values
        tokens = self.caption_model.generate(pixel_values, **self.gen_kwargs)
        caption = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return caption