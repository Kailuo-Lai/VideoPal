'''
Author: VideoPal Team
Date: 2024-04-10 13:44:55
LastEditors: VideoPal Team
LastEditTime: 2024-04-14 19:42:48
FilePath: /root/autodl-fs/projects/VideoPal/backend/backend_multimodal.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from sqlalchemy import and_
from typing import List
import time
import tqdm

from utils.database import VisualResult, AudioResult, VideoCapResult, VideoSummaryResult, VideoEventsSummaryResult, db
from utils.inputparser import TimeParser
from backend.backend_visual import VisualBackend
from models.llm_model import LLMModel
from models.kts_model import KTSModel

question_prompt_template = """
You are an expert in multimodal information merging, and you need to summarize and merge the information of the visual and auditory modalities of the video.\n
Visual information starts with: "The visual information you get is: "
Audio information starts with: "The auditory information you get is: "
You need to combine contextual, visual, and auditory information, make some imagination, 
and finally describe what is happening in the video clip, the more detailed the better.\n
Attention! You only need to answer descriptive text.\n
% USER QUERY:
{text}
YOUR RESPONSE:
"""

refine_prompt_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary\n"
    "If the context isn't useful, return the original summary."
)


class MultimodalBackend:
    def __init__(self,
                 kts_model: KTSModel,
                 llm_model: LLMModel,
                 args):
        '''
        Multimodal Backend for VideoPal
        kts_model: KTS Model
        llm_model: LLM Model
        args: Arguments
        '''
        self.args = args
        self.kts = kts_model
        self.llm = llm_model
        
        self.question_prompt = PromptTemplate(template=question_prompt_template, input_variables=['text'])
        self.refine_prompt = PromptTemplate(input_variables=["existing_answer", "text"], template=refine_prompt_template)

    @staticmethod
    def dynamic_split_segwindows(seg_windows: List[tuple], max_seconds: int = 300):
        '''
        Refine the seg_windows to make sure the length of each segment is less than max_seconds
        seg_windows: List of tuple, each tuple is a segment window e.g. (start, end)
        max_seconds: Maximum length of each segment
        '''
        new_seg_windows = []
        for window in seg_windows:
            if window[1] - window[0] > max_seconds:
                start = list(range(window[0], window[1], max_seconds))
                end = list(range(window[0] + max_seconds, window[1] + max_seconds, max_seconds))
                end[-1] = seg_windows[-1][1]
                interplolate_seg_windows = list(zip(start, end))
                new_seg_windows.extend(interplolate_seg_windows)
            else:
                new_seg_windows.append(window)
        return new_seg_windows
    
    def get_multimodal_summary(self, video_info):
        '''
        Get multimodal summary for a video
        video_info: [vid, video_name, video_length, filepath]
        '''
        print(f'\033[1;33mStarting Get Summary of Vid: {video_info[0]}, Video: {video_info[1]}\033[0m')
        start_time = time.perf_counter()
        seg_windows = VisualBackend.segment_video_(video_info, self.kts, 5)
        final_info = []
        nl = "\n"
        seg_windows = self.dynamic_split_segwindows(seg_windows, 300)
        print(f"Num of Segments: {len(seg_windows)}")
        for window in seg_windows:
            start, end = window[0], window[1]
            visual_results = db.query(VisualResult).filter(
            and_(
                VisualResult.vid == video_info[0],
                VisualResult.time >= start,
                VisualResult.time < end
                )
            ).all()

            video_cap_results = db.query(VideoCapResult).filter(
                and_(
                    VideoCapResult.vid == video_info[0],
                    VideoCapResult.start_time >= start,
                    VideoCapResult.end_time < end
                )
            ).all()

            audio_results = db.query(AudioResult).filter(
                and_(
                    AudioResult.vid == video_info[0],
                    AudioResult.start_time >= start,
                    AudioResult.end_time < end
                )
            ).all()
            
            visual_info, audio_info = [], []
            visual_info.extend([f"When {TimeParser.convert_time_format(item.time)}\nI saw {item.tag}\nI found {item.caption}\nI got subtitles {item.OCR}" for item in visual_results])
            visual_info.extend([f"When {TimeParser.convert_time_format(item.start_time)} - {TimeParser.convert_time_format(item.end_time)}, I saw '{item.content}'." for item in video_cap_results])
            audio_info.extend([f"When {TimeParser.convert_time_format(item.start_time)} - {TimeParser.convert_time_format(item.end_time)}, I heard '{item.content}'." for item in audio_results])
            final_info.append(
                Document(page_content = self.llm.check_and_sample_input(f"The visual information you get is:\n{nl.join(visual_info)}\nThe audio information you get is:\n{nl.join(audio_info)}").strip())
                )
            
        
        initial_chain = LLMChain(llm = self.llm.llm, prompt = self.question_prompt, llm_kwargs={"max_new_tokens": 500})
        refine_chain = LLMChain(llm = self.llm.llm, prompt = self.refine_prompt, llm_kwargs={"max_new_tokens": 500})
        refine_chain = RefineDocumentsChain(
            initial_llm_chain = initial_chain,
            refine_llm_chain = refine_chain,
            document_variable_name = "text",
            initial_response_name = "existing_answer"
            )
        refine_outputs = refine_chain(final_info)
        
        db.add(
            VideoSummaryResult(
                vid = video_info[0],
                video_name = video_info[1],
                content = refine_outputs['output_text']
                )
            )
        db.commit()
        print(f'\033[1;33mFinished After {time.perf_counter() - start_time} Seconds\033[0m')
        return refine_outputs
    
    def get_multimodal_event_summary(self, video_info):
        '''
        Get multimodal summary for events of a video
        video_info: [vid, video_name, video_length, filepath]
        '''
        print(f'\033[1;33mStarting Get Events Summary of Vid: {video_info[0]}, Video: {video_info[1]}\033[0m')
        start_time = time.perf_counter()
        seg_windows = VisualBackend.segment_video_(video_info, self.kts, 5)
        final_info = []
        nl = "\n"
        seg_windows = self.dynamic_split_segwindows(seg_windows, 300)
        print(f"Num of Segments: {len(seg_windows)}")
        for window in seg_windows:
            start, end = window[0], window[1]
            visual_results = db.query(VisualResult).filter(
            and_(
                VisualResult.vid == video_info[0],
                VisualResult.time >= start,
                VisualResult.time < end
                )
            ).all()

            video_cap_results = db.query(VideoCapResult).filter(
                and_(
                    VideoCapResult.vid == video_info[0],
                    VideoCapResult.start_time >= start,
                    VideoCapResult.end_time < end
                )
            ).all()

            audio_results = db.query(AudioResult).filter(
                and_(
                    AudioResult.vid == video_info[0],
                    AudioResult.start_time >= start,
                    AudioResult.end_time < end
                )
            ).all()
            
            visual_info, audio_info = [], []
            visual_info.extend([f"When {TimeParser.convert_time_format(item.time)}\nI saw {item.tag}\nI found {item.caption}\nI got subtitles {item.OCR}" for item in visual_results])
            visual_info.extend([f"When {TimeParser.convert_time_format(item.start_time)} - {TimeParser.convert_time_format(item.end_time)}, I saw '{item.content}'." for item in video_cap_results])
            audio_info.extend([f"When {TimeParser.convert_time_format(item.start_time)} - {TimeParser.convert_time_format(item.end_time)}, I heard '{item.content}'." for item in audio_results])
            final_info.append(
                Document(page_content = self.llm.check_and_sample_input(f"The visual information you get is:\n{nl.join(visual_info)}\nThe audio information you get is:\n{nl.join(audio_info)}").strip())
                )
        
        map_chain = LLMChain(llm = self.llm.llm, prompt = self.question_prompt,
                             llm_kwargs={"max_new_tokens": 500})
        for i in tqdm.tqdm(range(len(seg_windows))):
            db.add(
                VideoEventsSummaryResult(
                    vid = video_info[0],
                    video_name = video_info[1],
                    start_time = seg_windows[i][0],
                    end_time = seg_windows[i][1],
                    content = map_chain.run(final_info[i])
                )
            )
        db.commit()
        print(f'\033[1;33mFinished After {time.perf_counter() - start_time} Seconds\033[0m')