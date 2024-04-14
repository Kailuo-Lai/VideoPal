'''
Author: VideoPal Team
Date: 2024-03-12 10:40:23
LastEditors: VideoPal Team
LastEditTime: 2024-04-14 19:43:05
FilePath: /root/autodl-fs/projects/VideoPal/download_ckpt.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

from huggingface_hub import snapshot_download
import whisper

# Clip
snapshot_download(repo_id='openai/clip-vit-base-patch32',
                  local_dir="./checkpoints/clip-vit-base-patch32")

# LLM
snapshot_download(repo_id="THUDM/chatglm3-6b-32k",
                  local_dir="./checkpoints/chatglm3-6b-32k")

# Embeddings
snapshot_download(repo_id='intfloat/multilingual-e5-small',
                  local_dir="./checkpoints/multilingual-e5-small")

# Whisper
# large
model = whisper.load_model('large', download_root='./checkpoints/whisper-large')
# medium
model = whisper.load_model('medium', download_root='./checkpoints/whisper-medium')

# KeyBERT
snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir="./checkpoints/all-MiniLM-L6-v2")

# video caption 
snapshot_download(repo_id = 'MCG-NJU/videomae-base',
                  local_dir='./checkpoints/videomae-base')

snapshot_download(repo_id='Neleac/timesformer-gpt2-video-captioning',
                  local_dir='./checkpoints/timesformer-gpt2-video-captioning')

# Translate
snapshot_download(repo_id = "Helsinki-NLP/opus-mt-zh-en", local_dir = "./checkpoints/Helsinki-NLP-opus-mt-zh-en")