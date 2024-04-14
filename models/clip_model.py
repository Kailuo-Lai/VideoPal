'''
Author: VideoPal Team
Date: 2024-03-02 11:20:30
LastEditors: VideoPal Team
LastEditTime: 2024-04-11 22:22:06
FilePath: /root/autodl-fs/projects/VideoPal/models/clip_model.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import cv2
import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, logging, CLIPTextModelWithProjection
from ipex_llm.optimize import load_low_bit, optimize_model
from utils.utils import new_cd
logging.set_verbosity_error()

parent_dir = os.path.dirname(__file__)

class CLIPModel():
    def __init__(self, args):
        with new_cd(parent_dir):
            self.processor = CLIPProcessor.from_pretrained(f"../checkpoints/{args.clip_version}")
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(f"../checkpoints/{args.clip_version}")
            self.text_model = CLIPTextModelWithProjection.from_pretrained(f"../checkpoints/{args.clip_version}")
            if args.clip_low_bit:
                self.vision_model = optimize_model(self.vision_model,low_bit='sym_int4')
                self.text_model = optimize_model(self.text_model,low_bit='sym_int4')