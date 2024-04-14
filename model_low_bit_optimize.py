'''
Author: VideoPal Team
Date: 2024-04-10 20:20:24
LastEditors: VideoPal Team
LastEditTime: 2024-04-10 20:20:28
FilePath: /root/autodl-fs/projects/VideoPal/model_low_bit_optimize.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import whisper
from ipex_llm import optimize_model
import os

# LLM
# ChatGLM3-6b-32k
print("\033[1;32mOptimize LLM...\033[0m")
model = AutoModel.from_pretrained('./checkpoints/chatglm3-6b-32k',
                                    load_in_low_bit="sym_int8",
                                    trust_remote_code=True)
model.save_low_bit('./checkpoints/chatglm3-6b-32k-optimized')
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/chatglm3-6b-32k',
                                            trust_remote_code=True)
tokenizer.save_pretrained('./checkpoints/chatglm3-6b-32k-optimized')

# Whisper
# large
print("\033[1;32mOptimize Whisper...\033[0m")
model_state_file = os.listdir(f"./checkpoints/whisper-large")[0]
model = whisper.load_model(f"./checkpoints/whisper-large/{model_state_file}", device="cpu")
model = optimize_model(model, low_bit='sym_int4')
model.save_low_bit("./checkpoints/whisper-large-optimized")

# medium
model_state_file = os.listdir(f"./checkpoints/whisper-medium")[0]
model = whisper.load_model(f"./checkpoints/whisper-medium/{model_state_file}", device="cpu")
model = optimize_model(model, low_bit='sym_int4')
model.save_low_bit("./checkpoints/whisper-medium-optimized")

# CLIP vision part
print("\033[1;32mOptimize CLIP...\033[0m")
model = CLIPVisionModelWithProjection.from_pretrained("./checkpoints/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained(f"./checkpoints/clip-vit-base-patch32")
model = optimize_model(model, low_bit='sym_int4')
model.save_low_bit("./checkpoints/clip-vit-base-patch32-vision-optimized")
processor.save_pretrained("./checkpoints/clip-vit-base-patch32-vision-optimized")

# CLIP text part
model = CLIPTextModelWithProjection.from_pretrained("./checkpoints/clip-vit-base-patch32")
model = optimize_model(model, low_bit='sym_int4')
model.save_low_bit("./checkpoints/clip-vit-base-patch32-text-optimized")