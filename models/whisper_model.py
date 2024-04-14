'''
Author: VideoPal Team
Date: 2024-03-02 11:20:30
LastEditors: VideoPal Team
LastEditTime: 2024-03-24 12:58:07
FilePath: /chengruilai/projects/VideoPal/models/whisper_model.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import os
import whisper
from ipex_llm.optimize import load_low_bit
from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)

class WhisperModel():
    def __init__(self, args):
        with new_cd(parent_dir):
            model_state_file = os.listdir(f"../checkpoints/whisper-{args.whisper_version}")[0]
            self.model = whisper.load_model(f"../checkpoints/whisper-{args.whisper_version}/{model_state_file}")
            if args.whisper_low_bit:
                self.model = load_low_bit(self.model, f"../checkpoints/whisper-{args.whisper_version}-optimized")