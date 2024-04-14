'''
Author: VideoPal Team
Date: 2024-03-23 23:36:53
LastEditors: VideoPal Team
LastEditTime: 2024-03-24 23:06:40
FilePath: /chengruilai/projects/VideoPal/models/translate_model.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.utils import new_cd
parent_dir = os.path.dirname(__file__)

class Translator:
    def __init__(self, args) -> None:
        self.args = args
        with new_cd(parent_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(f"../checkpoints/Helsinki-NLP-opus-mt-{self.args.convert_lid}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"../checkpoints/Helsinki-NLP-opus-mt-{self.args.convert_lid}")
    def __call__(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids=input_ids)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return outputs