'''
Author: VideoPal Team
Date: 2024-03-02 15:08:52
LastEditors: VideoPal Team
LastEditTime: 2024-04-14 21:46:31
FilePath: /root/autodl-fs/projects/VideoPal/main_gradio.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import time
import gradio as gr
import pandas as pd
import argparse
from paddleocr import PaddleOCR
from typing import Union
import os
import shutil
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['NUMEXPR_MAX_THREADS'] = '128'

from models import whisper_model, clip_model, kts_model, tag2text_model, llm_model, keybert_model, video_caption_model, text_embedding_model, \
    translate_model
from backend.backend_audio import AudioBackend
from backend.backend_visual import VisualBackend
from backend.backend_multimodal import MultimodalBackend
from backend.backend_search import SearchBackend
from utils.database import db, VideoInfo, VideoResultComplete, AudioResult, VideoCapResult, VideoSummaryResult, VideoEventsSummaryResult, VisualResult
from utils.utils import get_video_info

parser = argparse.ArgumentParser()

# kts arguments
parser.add_argument("--alpha", default=20, type=int, help="Determine the maximum segment number for KTS algorithm, the larger the value, the fewer segments.")
parser.add_argument("--vmax", default=0.3, type=float, help="Special parameter of penalty term for KTS algorithm, the larger the value, the fewer segments.")

# clip model arguments
parser.add_argument("--clip_version", default="clip-vit-base-patch32", help="Clip model version for video feature extractor")
parser.add_argument("--clip_low_bit", default=False, action="store_true", help="load clip low bit model or not")

# tag2text model arguments
parser.add_argument("--tag2text_thershld", default=0.68, type=float, help="Threshold for tag2text model")

# whisper model arguments
parser.add_argument("--whisper_version", default="medium", help="Whisper model version for video asr")
parser.add_argument("--whisper_low_bit", default=False, action="store_true", help="load whisper low bit model or not")

# llm model arguments
parser.add_argument("--llm_version", default="chatglm3-6b-32k-optimized", help="LLM model version")
parser.add_argument("--max_tokens", default=5000, type=int, help="Maximum tokens for LLM model")
parser.add_argument("--max_length", default=5000, type=int, help="Maximum length for LLM model")

# KeyBert model arguments
parser.add_argument("--keybert_version", default="all-MiniLM-L6-v2", help="KeyBert model version")

# video caption model arguments
parser.add_argument("--num_beams", default=6, help="Number of frames for video caption")
parser.add_argument("--video_caption_low_bit", default=False, action="store_true", help="load video caption model low bit model or not")

# text embedding arguments
parser.add_argument("--embed_version", default="multilingual-e5-small", help="Embedding model version")
parser.add_argument("--embed_low_bit", default=True, help="load Embedding model low bit model or not")

# translate model arguments
parser.add_argument("--convert_lid", default="zh-en", help="convert language id")

# visual backend arguments
parser.add_argument("--inter_seconds", default=5, type=int, help="The smallest time gap between successive clips, in seconds for static information.")
parser.add_argument("--inter_seconds_video_cap", default=20, help="The smallest time gap between successive clips, in seconds for video caption.")

# general arguments
parser.add_argument("--port", type = int, default = 8899, help = "Gradio server port")
parser.add_argument("--mode", default="normal", choices=["debug", "normal"], help="Which mode do you want to run, debug or normal. Debug: More detailed output")

args = parser.parse_args()
print(args)

video_info = [None, None, None, None]
global_chat_history = []

print('\033[1;32m' + "Initializing models...".center(50, '-') + '\033[0m')
start_time = time.perf_counter()
whisper = whisper_model.WhisperModel(args)
clip = clip_model.CLIPModel(args)
kts = kts_model.KTSModel(args)
tag2text = tag2text_model.Tag2TextModel(args)
kerbert = keybert_model.KeyBertModel(args)
text_emb = text_embedding_model.TextEmbedding(args)
llm = llm_model.LLMModel(args)
ocr = PaddleOCR(use_angle_cls = True, lang = "ch", show_log = False)
translator = translate_model.Translator(args)
videocap = video_caption_model.VideoCaptionModel(args)
print(f"\033[1;32mModel initialization finished after {time.perf_counter() - start_time}s".center(50, '-') + '\033[0m')

audio_backend = AudioBackend(whisper, args)
visual_backend = VisualBackend(clip, tag2text, kts, ocr, videocap, translator, args)
multimodal_backend = MultimodalBackend(kts, llm, args)
search_backend = SearchBackend(clip, llm, kerbert, text_emb, translator,  video_info, args)

# demo1
def show_example():
    video_inp = './video_storage/example_贝爷求生.mp4'
    global video_info
    video_name = video_inp.split("/")[-1]
    data = db.query(VideoInfo).filter(VideoInfo.video_name == video_name).first()
    if data != None:
        raise gr.Error(f"Vid: {data.vid}, Video {video_name} already exists in database")
    else:
        print(f"\033[36;1mUpload {video_name} into Database\033[0m")
        video_info = get_video_info(f"./video_storage/{video_name}", db)
        search_backend.video_info = video_info
        # print(f"\033[36;1mCurrent Video Info {video_info}\033[0m")
        return gr.update(value = f"vid: {video_info[0]} | video name: {video_info[1]} | length: {round(video_info[2], 2)}s"), gr.update(value = "No"), gr.update(value = video_info[3])

def video_inp_preprocess(video_inp: str):
    global video_info
    video_name = video_inp.split("/")[-1]
    data = db.query(VideoInfo).filter(VideoInfo.video_name == video_name).first()
    if data != None:
        raise gr.Error(f"Vid: {data.vid}, Video {video_name} already exists in database")
    else:
        print(f"\033[36;1mUpload {video_name} into Database\033[0m")
        if os.path.exists(f"./video_storage/{video_name}"):
            os.remove(f"./video_storage/{video_name}")
        shutil.move(video_inp, "./video_storage/")
        video_info = get_video_info(f"./video_storage/{video_name}", db)
        search_backend.video_info = video_info
        # print(f"\033[36;1mCurrent Video Info {video_info}\033[0m")
        return gr.update(value = f"vid: {video_info[0]} | video name: {video_info[1]} | length: {round(video_info[2], 2)}s"), gr.update(value = "No"), gr.update(value = video_info[3])

def check_video_result_state(filepath: str):
    video_name = filepath.split("/")[-1]
    if db.query(VideoResultComplete).filter(VideoResultComplete.video_name == video_name).first() == None:
        return False
    else:
        return True
        
def change_cur_video_by_id_or_name(video_id: Union[int, str]):
    global video_info
    if video_id.isdigit():
        video_id = int(video_id)
    if isinstance(video_id, int):
        data = db.query(VideoInfo).filter(VideoInfo.vid == video_id).first()
        if data == None:
            raise gr.Error(f"No video with vid {video_id} in database")
        else:
            video_info = [data.vid, data.video_name, data.video_length, data.filepath]
            search_backend.video_info = video_info
            print(f"\033[36;1mChange to Video: {data.video_name}, Vid: {data.vid}\033[0m")
            if check_video_result_state(data.filepath):
                return gr.update(value = f"vid: {video_info[0]} | video name: {video_info[1]} | length: {round(video_info[2], 2)}s"), gr.update(value = video_info[3]), gr.update(value = "Yes")
            else:
                gr.Warning(f"Video {video_info[1]} hasn't completed information extraction.")
                return gr.update(value = f"vid: {video_info[0]} | video name: {video_info[1]} | length: {round(video_info[2], 2)}s"), gr.update(value = video_info[3]), gr.update(value = "No")
    elif isinstance(video_id, str):
        data = db.query(VideoInfo).filter(VideoInfo.video_name == video_id).first()
        if data == None:
            raise gr.Error(f"No video with name {video_id} in database")
        else:
            video_info = [data.vid, data.video_name, data.video_length, data.filepath]
            search_backend.video_info = video_info
            print(f"\033[36;1mChange to Video: {data.video_name}, Vid: {data.vid}\033[0m")
            if check_video_result_state(data.filepath):
                return gr.update(value = f"vid: {video_info[0]} | video name: {video_info[1]} | length: {round(video_info[2], 2)}s"), gr.update(value = video_info[3]), gr.update(value = "Yes")
            else:
                gr.Warning(f"Video {video_info[1]} hasn't completed information extraction.")
                return gr.update(value = f"vid: {video_info[0]} | video name: {video_info[1]} | length: {round(video_info[2], 2)}s"), gr.update(value = video_info[3]), gr.update(value = "No")
        
def extract_information_from_video(reload_flag: str):
    print(f"\033[36;1mCurrent Video Info {video_info}\033[0m")
    filepath = video_info[3]
    
    if check_video_result_state(filepath):
        if reload_flag:
            gr.Warning(f"Video {filepath.split('/')[-1]} already has information extracted, we will re-extract and overwrite the previous data.")
            
            data = db.query(VideoResultComplete).filter(VideoResultComplete.video_name == filepath.split("/")[-1]).delete()
            
            data = db.query(AudioResult).filter(AudioResult.video_name == filepath.split("/")[-1]).delete()
            data = db.query(VisualResult).filter(VisualResult.video_name == filepath.split("/")[-1]).delete()
            data = db.query(VideoCapResult).filter(VideoCapResult.video_name == filepath.split("/")[-1]).delete()
            
            data = db.query(VideoSummaryResult).filter(VideoSummaryResult.video_name == filepath.split("/")[-1]).delete()
            data = db.query(VideoEventsSummaryResult).filter(VideoEventsSummaryResult.video_name == filepath.split("/")[-1]).delete()
            
            db.commit()
        else:
            raise gr.Error(f"Video {filepath.split('/')[-1]} already has information extracted! If you want to re-extract, please click the button again and set the reload flag to True")
        
    data = db.query(VideoInfo).filter(VideoInfo.video_name == filepath.split("/")[-1]).first()
    vid, video_name, video_length, filepath  = data.vid, data.video_name, data.video_length, data.filepath
    print(f"\033[36;1mStarting Information Extraction Process of Video: {video_name}, Vid: {vid}\033[0m")
    audio_backend.get_audio_info(filepath, video_info)
    visual_backend.get_static_info_of_whole_video(filepath, video_info)
    visual_backend.get_dynamic_info_of_whole_video(filepath, video_info)
    
    multimodal_backend.get_multimodal_summary(video_info)
    multimodal_backend.get_multimodal_event_summary(video_info)
    db.add(VideoResultComplete(vid = vid, video_name = video_name, video_length = video_length, filepath = filepath))
    db.commit()
    print(f"\033[36;1mInformation Extraction Process of Video: {video_name}, Vid: {vid} Completed\033[0m")
    return gr.update(value = "Yes")

def clean_conversation():
    global global_chat_history
    global_chat_history = []
    print(f"\033[36;1mClean Conversation...\033[0m")
    return gr.update(value = None), gr.update(value = None), gr.update(value = None), gr.update(value = None), gr.update(value = None), gr.update(value = None), gr.update(value = None)

def clean_chat_history():
    global global_chat_history
    global_chat_history = []
    print(f"\033[36;1mClean Chat History...\033[0m")
    return gr.update(value = None), gr.update(value = None), gr.update(value = None)

def submit_message(query):
    data = db.query(VideoResultComplete).filter(VideoResultComplete.vid == video_info[0]).first()
    if data == None:
        raise gr.Error(f"Video {video_info[1]} hasn't completed information extraction.")
    print(f"\033[36;1mQuerying...\033[0m")
    response = search_backend.get_final_answer(query)
    global_chat_history.append((query, response['output']))
    intermidiate_msg = f"Tools:\n{response['tool']}\nObservation:\n{response['observation']}\nOutput:\n{response['output']}"
    return global_chat_history, gr.update(value = intermidiate_msg)

# demo2
def renew_video_info():
    print(f"\033[36;1mRenew Video Files Preview...\033[0m")
    video_info_data = pd.read_sql(db.query(VideoInfo).statement, db.bind)
    video_info_complete_data = pd.read_sql(db.query(VideoResultComplete).statement, db.bind)
    return gr.update(value = video_info_data), gr.update(value = video_info_complete_data)


    


css = """
      #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
      #video_inp {min-height: 100px}
      #chatbox {min-height: 400px;}
      #header {text-align: center;}
      #hint {font-size: 1.0em; padding: 0.5em; margin: 0;}
      .message { font-size: 1.2em; }
      """      
    

with gr.Blocks(css=css) as demo1:
    
    with gr.Column(elem_id = "col-container"):
        gr.Markdown("""
                    <h1 align="center"><img src="file/VideoPal_logo.jpeg", border="0" style="margin: 0 auto; height: 200px;"/></h1>
                    <h1 align="center">🐿️VideoPal: Pal for Long Video Chat</h1>
                    <h5 align="center">Powered by BigDL, ChatGLM3, CLIP, Whisper, Tag2Text, KeyBert, Video Caption Model etc.</h5> 
                    """)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    btn_show_example = gr.Button("Show Example")
                    reload_flag = gr.Checkbox(label="Reload", info="Whether to reload the video information extraction process if the video has already been processed.")
                with gr.Row():
                    cur_video_info = gr.Textbox(label="Current video", interactive=False, value="")
                    video_state = gr.Textbox(label = "Completion of Information Extraction", interactive=False, value="")
                video_inp = gr.Video(label="Upload Video", elem_id="video_inp", source='upload')
                video_information_extraction_btn = gr.Button("Video Information Extraction")
                with gr.Row():
                    vid_change = gr.Textbox(value="", placeholder="vid or video name", show_label=False)
                    video_change_btn = gr.Button("Change Video")

            with gr.Column():
                chatbot = gr.Chatbot(elem_id="chatbox")
                input_msg = gr.Textbox(show_label=False, placeholder="Enter text and press enter", visible=True).style(container=False)
                btn_submit = gr.Button("Submit")
                with gr.Row():
                    btn_clean_chat_history = gr.Button("Clean Chat History")
                    btn_clean_conversation = gr.Button("🔃 Start New Conversation")
        agent_msg = gr.Textbox(label="Agent Intermediate Message", interactive=False, value="", lines=20)
    
    btn_show_example.click(show_example, [], [cur_video_info, video_state, video_inp])
    video_inp.upload(video_inp_preprocess, [video_inp], [cur_video_info, video_state, video_inp])
    video_information_extraction_btn.click(extract_information_from_video, [reload_flag], [video_state])
    video_change_btn.click(change_cur_video_by_id_or_name, [vid_change], [cur_video_info, video_inp, video_state])
    
    btn_clean_chat_history.click(clean_chat_history, [], [input_msg, chatbot, agent_msg])
    btn_clean_conversation.click(clean_conversation, [], [cur_video_info, video_state, video_inp, vid_change, chatbot, input_msg, agent_msg])
    
    btn_submit.click(submit_message, [input_msg], [chatbot, agent_msg])
    

with gr.Blocks(css=css) as demo2:
    with gr.Column(elem_id = "col-container"):
        with gr.Row():
            video_info_dataframe = gr.DataFrame(interactive=False, label="Video Info (Database)")
            video_result_complete_dataframe = gr.DataFrame(interactive=False, label="Video Info (Information Extraction Process Complete)")
        
    video_info_update_btn = gr.Button(value = "Update Video Info")
    
    video_info_update_btn.click(renew_video_info, [], [video_info_dataframe, video_result_complete_dataframe])

app = gr.TabbedInterface([demo1, demo2], ["VideoPal", "Files"])
app.launch(share = False, debug = True, server_port = args.port)