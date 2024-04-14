'''
Author: VideoPal Team
Date: 2024-03-10 12:34:10
LastEditors: Kailuo_Lai 1090087070@qq.com
LastEditTime: 2024-04-14 22:43:25
FilePath: /root/autodl-fs/projects/VideoPal/backend/backend_search.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import torch
from langchain import  LLMChain
from langchain.agents import Tool, AgentExecutor
from typing import List
from sqlalchemy import and_

from models.clip_model import CLIPModel
from models.keybert_model import KeyBertModel
from models.text_embedding_model import TextEmbedding
from models.translate_model import Translator
from models.llm_model import LLMModel

from utils.inputparser import TimeParser, translate_zh_en 
from utils.database import VisualResult, AudioResult, VideoCapResult, VideoSummaryResult, VideoEventsSummaryResult, db
from utils.searchtools import cosine_similarity, get_clip_embedding
from utils.llm_agent import CustomPromptTemplate, CustomOutputParser, agent_template, LLMSingleActionAgent


class SearchBackend:
    def __init__(self, 
                 clip_model: CLIPModel, 
                 llm_model: LLMModel, 
                 keybert_model: KeyBertModel,
                 text_emb_model: TextEmbedding,
                 translator: Translator,
                 video_info: List,
                 args) -> None:
        '''
        Search backend for VideoPal.
        clip_model: CLIP model
        llm_model: LLM model
        keybert_model: KeyBERT model
        text_model: TextEmbedding model
        translator: Translator
         video_info: [vid, video_name, video_length, filepath]
        args: Arguments
        '''
        self.clip_model = clip_model
        self.keybert_model = keybert_model
        self.text_emb_model = text_emb_model
        self.args = args
        self.llm = llm_model
        self.video_info = video_info
        self.translator = translator
        
    def get_visual_info_from_time(self, video_info: List, start_time: int, end_time: int) -> list:
        '''
        Get visual information based on time
         video_info: [vid, video_name, video_length, filepath]
        start_time: Start time of the interval
        end_time: End time of the interval
        '''
        # Query the database for records where 'time' is between start_time and end_time
        results = db.query(VisualResult).filter(
            and_(
                VisualResult.vid == video_info[0],
                VisualResult.time >= start_time,
                VisualResult.time <= end_time
            )
        ).all()

        # Format the output according to "when {time}, i saw {tag} i found {caption}"
        visual_info = [
            f"When {TimeParser.convert_time_format(result.time)}\nI saw {result.tag}\nI found {result.caption}\nI got subtitles {result.OCR}"
            for result in results
        ]

        return visual_info
    
    def get_visual_info_from_entity(self, video_info: List, entity: str) -> list:
        '''
        Get the visual information based on entity
         video_info: [vid, video_name, video_length, filepath]
        entity: Entity to search for
        '''
        # Query the database for all records
        results = db.query(VisualResult).filter(VisualResult.vid == video_info[0]).all()

        # Convert database results to a format similar to the original CSV for processing
        data = [{
            'frame_time': result.time,
            'tags': result.tag,
            'caption': result.caption,
            'OCR':result.OCR,
            'clip_features': get_clip_embedding(video_info)[result.clip_id]
        } for result in results]

        # Process entity through CLIP model for embedding
        entity_input = self.clip_model.processor(text = entity, return_tensors = "pt", padding = True, truncation = True, max_length = 50)
        with torch.inference_mode():
            entity_embedding = self.clip_model.text_model(**entity_input)['text_embeds'].numpy()

        # Calculate the cosine similarity for each row, this part needs to work with numpy arrays and might require fetching data differently
        for item in data:
            item['similarity'] = cosine_similarity([item['clip_features']], entity_embedding[0])

        # Sort data by similarity and filter
        data.sort(key=lambda x: x['similarity'], reverse=True)
        top_similar_rows = data[:10]  # Assuming you want top 10 similar

        # Filter rows where 'tags' or 'caption' contain the entity, or they are in top similar rows
        filtered_data = [
            item for item in data if (
                entity.lower() in item['tags'].lower() or 
                entity.lower() in item['caption'].lower() or 
                entity.lower() in item['OCR'].lower() or
                item in top_similar_rows
            )
        ]

        # Format the output
        visual_info = []
        for item in filtered_data:
            captions = self.get_video_info_from_time(video_info, item['frame_time'] - 5, item['frame_time'] + 5)
            visual_info.append(f"When {TimeParser.convert_time_format(item['frame_time'])}\nI saw {item['tags']}\nI found {item['caption']}\nI got subtitles {item['OCR']}")
            visual_info.append('\n'.join(captions))
        
        
        return visual_info
    
    def get_audio_info_from_time(self, video_info: List, start_time:int, end_time:int) -> list:
        '''
        Get audio information based on time
         video_info: [vid, video_name, video_length, filepath]
        start_time: Start time of the interval
        end_time: End time of the interval
        '''
        def format_audio_info(result: AudioResult):
            return f"When {TimeParser.convert_time_format(result.start_time)} - {TimeParser.convert_time_format(result.end_time)}, I heard '{result.content}'."
        # Query the database for records where 'start_time' and 'end_time' overlap with the given interval
        results = db.query(AudioResult).filter(
            and_(
                AudioResult.vid == video_info[0],
                AudioResult.start_time <= end_time,
                AudioResult.end_time >= start_time
            )
        ).all()

        # Format the output according to "when {start_time} - {end_time}, i heard '{content}'"
        audio_info = [
            format_audio_info(result)
            for result in results
        ]

        return audio_info
    

    def get_audio_info_from_entity(self, video_info: List, entity:str) -> list:
        '''
        Get audio information based on entity
         video_info: [vid, video_name, video_length, filepath]
        entity: Entity to search for
        '''
        # Define a helper function to format the audio information string
        def format_audio_info(result: AudioResult):
            return f"When {TimeParser.convert_time_format(result.start_time)} - {TimeParser.convert_time_format(result.end_time)}, I heard '{result.content}'."

        # Initialize an empty list to store the formatted audio information
        audio_info = []

        # Query the database for all records
        results = db.query(AudioResult).filter(AudioResult.vid == video_info[0]).order_by(AudioResult.id).all()

        # Iterate over the query results
        for index, result in enumerate(results):
            # Check if 'content' contains the entity
            if entity.lower() in result.content.lower():
                # Append the previous row if it exists
                if index > 0:
                    prev_result = results[index - 1]
                    audio_info.append(format_audio_info(prev_result))
                
                # Append the current row
                audio_info.append(format_audio_info(result))

                # Append the next row if it exists
                if index < len(results) - 1:
                    next_result = results[index + 1]
                    audio_info.append(format_audio_info(next_result))

        audio_contents = [line.content for line in results]
        with torch.no_grad():
            entity_embedding = self.text_emb_model([entity])
            audio_embedding = self.text_emb_model(audio_contents)
            topsims, topidxs = torch.cosine_similarity(entity_embedding,audio_embedding).topk(3)
        topidxs = topidxs.tolist()
        topsims = topsims.tolist()
        for idx, sim in zip(topidxs,topsims):
            info = format_audio_info(results[idx])
            if info not in audio_info and sim > 0.95:
                audio_info.append(info)
        
        return audio_info
    
    def get_video_info_from_time(self, video_info: List, start_time:int, end_time:int) -> list:
        '''
        Get video caption information based on time
         video_info: [vid, video_name, video_length, filepath]
        start_time: Start time of the interval
        end_time: End time of the interval
        '''
        # Query the database for records where 'start_time' and 'end_time' overlap with the given interval
        results = db.query(VideoCapResult).filter(
            and_(
                VideoCapResult.vid == video_info[0],
                VideoCapResult.start_time <= end_time,
                VideoCapResult.end_time >= start_time
            )
        ).all()

        # Format the output according to "when {start_time} - {end_time}, i heard '{content}'"
        video_cap_info = [
            f"When {TimeParser.convert_time_format(result.start_time)} - {TimeParser.convert_time_format(result.end_time)}, I saw '{result.content}'."
            for result in results
        ]

        return video_cap_info

    def get_information_from_detailed_objects(self, input: str) -> str:
        '''
        Get [visual, audio, video caption] information based on entity
        input: User query
        '''
        nl = '\n'
        non_time_entity =  self.keybert_model.extract_keywords(input)
        if len(non_time_entity) != 0:
            visual_info = []
            audio_info = []
            if isinstance(non_time_entity, list):
                for entity in non_time_entity:
                    
                    visual_info.extend(self.get_visual_info_from_entity(self.video_info, entity))
                    audio_info.extend(self.get_audio_info_from_entity(self.video_info, entity))
                    
                final_info = f"The visual information you get is:\n{nl.join(visual_info)}\nThe audio information you get is:\n{nl.join(audio_info)}"
                
                return self.llm.check_and_sample_input(final_info)
            else:
                raise ValueError("Invalid non-time entity format")
        else:
            # Visual information processing
            visual_results = db.query(VisualResult).filter(
                VisualResult.vid == self.video_info[0]
            ).all()
            visual_info = [
                f"When {TimeParser.convert_time_format(result.time)}\nI saw {result.tag}\nI found {result.caption}\nI got subtitles {result.OCR}"
                for result in visual_results
            ]

            # Video caption information processing
            video_results = db.query(VideoCapResult).filter(
                VideoCapResult.vid == self.video_info[0]
            ).all()
            video_cap_info = [
                f"When {TimeParser.convert_time_format(result.start_time)} - {TimeParser.convert_time_format(result.end_time)}, I saw '{result.content}'."
                for result in video_results
            ]

            # Audio information processing
            audio_results = db.query(AudioResult).filter(
                AudioResult.vid == self.video_info[0]
            ).all()
            audio_info = [
                f"When {TimeParser.convert_time_format(result.start_time)} - {TimeParser.convert_time_format(result.end_time)}, I heard '{result.content}'."
                for result in audio_results
            ]

            # Combining all the formatted information
            final_info = f"The visual information you get is:\n{nl.join(visual_info)}\n{nl.join(video_cap_info)}\nThe audio information you get is:\n{nl.join(audio_info)}"

            # You can return this final_info or use it as needed within the function
            return self.llm.check_and_sample_input(final_info)
            
        
    def get_information_from_detailed_time(self, input: str) -> str:
        '''
        Get [visual, audio, video caption] information based on time
        input: User query
        '''
        nl = '\n'
        [start_time, end_time] = list(self.llm.get_time_entity_from_query(input).values())
        start_time, end_time = TimeParser.time_to_seconds(start_time), TimeParser.time_to_seconds(end_time)
        visual_info = self.get_visual_info_from_time(self.video_info, start_time, end_time)
        audio_info = self.get_audio_info_from_time(self.video_info, start_time, end_time)
        video_cap_info = self.get_video_info_from_time(self.video_info, start_time, end_time)
        visual_info.extend(video_cap_info)
        final_info = f"The visual information you get is:\n{nl.join(visual_info)}\nThe audio information you get is:\n{nl.join(audio_info)}"
    
        return self.llm.check_and_sample_input(final_info)
    
    def get_information_from_video_summary(self, input: str) -> str:
        '''
        Get video summary based on entity
        input: User query
        '''
        summary_results = db.query(VideoSummaryResult).filter(VideoSummaryResult.vid == self.video_info[0]).all()
        event_summary_results = db.query(VideoEventsSummaryResult).filter(VideoEventsSummaryResult.vid == self.video_info[0]).all()
        
        summary_info = [f"Summary of the entire video:\n{summary_results[0].content}\nSummary of the clips:"] + \
            [
                f"When {TimeParser.convert_time_format(result.start_time)} - {TimeParser.convert_time_format(result.end_time)}, {result.content}"
                for result in event_summary_results
            ]
        return '\n'.join(summary_info)

    def get_final_answer(self, query: str):     
        query = translate_zh_en(query, self.translator)
        tools = [
            Tool(name="get_information_from_detailed_time", func=self.get_information_from_detailed_time, description="if you has a detailed time or time range, the tool can return the information of the time or time range"),
            Tool(name="get_information_from_detailed_objects", func=self.get_information_from_detailed_objects, description="if you has a detailed non time entity(detailed object/scene/person), the tool can return the detailed time or time range of the entity"),
            Tool(name="get_information_from_video_summary", func=self.get_information_from_video_summary, description="get summary of the video")
        ]
        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser()
        prompt = CustomPromptTemplate(template = agent_template,
                                      tools = tools,
                                      input_variables = ["query", "intermediate_steps"])
        llm_chain = LLMChain(llm = self.llm.llm, prompt = prompt, llm_kwargs={"max_new_tokens": 500})
        agent = LLMSingleActionAgent(
            llm_chain = llm_chain,
            output_parser = output_parser,
            stop = ["\nObservation:"],
            allowed_tools = tool_names,
        )
        agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True, return_intermediate_steps = True)
        response_steps = []
        for step in agent_executor.iter(inputs={"query": query}):  
            response_steps.append(step)
        extracted_info = []
        for entry in response_steps:
            item_info = {}
            steps_key = 'intermediate_step' if 'intermediate_step' in entry else 'intermediate_steps'
            if steps_key in entry:
                first_step = entry[steps_key][0]  
                item_info['input'] = first_step[0].tool_input  
                item_info['tool'] = first_step[0].tool  
                item_info['observation'] = first_step[1]  
            if 'output' in entry:
                item_info['output'] = entry['output']  
            extracted_info.append(item_info)
        return extracted_info[-1]