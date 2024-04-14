'''
Author: VideoPal Team
Date: 2024-03-02 11:20:30
LastEditors: VideoPal Team
LastEditTime: 2024-04-14 18:07:28
FilePath: /root/autodl-fs/projects/VideoPal/models/llm_model.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import os
from ipex_llm.langchain.llms import TransformersLLM
from ipex_llm.transformers import AutoModel
from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from transformers import AutoTokenizer
from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)


class LLMModel():
    def __init__(self, args):
        self.history = []
        self.llm_version = args.llm_version
        self.max_length = args.max_length
        self.max_tokens = args.max_tokens
        self.init_model()
    
    def init_model(self):
        with new_cd(parent_dir):
            self.llm = TransformersLLM.from_model_id_low_bit(f"../checkpoints/{self.llm_version}",
                                                             {"trust_remote_code": True, 
                                                              "max_length": self.max_length
                                                            #   "max_new_tokens": self.max_length
                                                              })
            self.tokenizer = self.llm.tokenizer
            self.llm.streaming = False

    def get_token_count(self, text):
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)
    
    def dynamic_sampling(self, lines, max_tokens):
        # Sampling so that the length of a input text does not exceed the maximum number of tokens
        step = 1  
        sampled_lines = lines[::step]
        sampled_text = '\n'.join(sampled_lines)
        token_count = self.get_token_count(sampled_text)

        # Increase the sampling interval until the sample size is less than the maximum number of tokens
        while token_count > max_tokens and step < len(lines):
            step += 1
            sampled_lines = lines[::step]
            sampled_text = '\n'.join(sampled_lines)
            token_count = self.get_token_count(sampled_text)
        return sampled_lines
    
    def check_and_sample_input(self, input_text):
        nl = '\n'
        # Split visual and audio information
        visual_text, audio_text = input_text.split('The audio information you get is:')
        visual_lines = visual_text.strip().split('\n')
        audio_lines = audio_text.strip().split('\n')

        # Calculate the number of tokens for visual and audio information
        visual_tokens_count = self.get_token_count(visual_text)
        audio_tokens_count = self.get_token_count(audio_text)
        total_tokens_count = visual_tokens_count + audio_tokens_count

        # Allocate the number of tokens proportionally
        if total_tokens_count > 0:  # Prevent division by zero
            max_visual_tokens = int((visual_tokens_count / total_tokens_count) * self.max_tokens)
            max_audio_tokens = self.max_tokens - max_visual_tokens
        else:
            # Prevent errors if input is empty
            max_visual_tokens = max_audio_tokens = self.max_tokens // 2

        sampled_visual_lines = self.dynamic_sampling(visual_lines, max_visual_tokens)
        sampled_audio_lines = self.dynamic_sampling(audio_lines, max_audio_tokens)

        # Combining the sampled text
        combined_sampled_text = f"{nl.join(sampled_visual_lines)}\nThe audio information you get is:\n{nl.join(sampled_audio_lines)}"

        return combined_sampled_text

    def get_time_entity_from_query(self, query: str):
            response_schemas = [
                ResponseSchema(name="start_time", description="Find the starting time in user query, and you must return the start time in the format of 'MM:SS' or 'SS', 'MM' means minutes, 'SS' means seconds."),
                ResponseSchema(name="end_time", description="Find the ending time in user query, and you must return the end time in the format of 'MM:SS' or 'SS', 'MM' means minutes, 'SS' means seconds."),
            ]

            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

            format_instructions = output_parser.get_format_instructions()

            template = (
                """
                You are a time expert, proficient in extracting time from user's query.

                You need to extract the start time and end time from user's query.
                If the start time is equal to the end time in user's query, please return the same time for both start time and end time.

                Examples of user queries and expected outputs:
                - Query: "From 34 minutes and 23 seconds to 34 minutes and 32 seconds, what are the characters in the video doing?"
                Expected Output: start_time: 34:23, end_time: 34:32
                - Query: "What object appears in the video at 1 minute and 32 seconds?"
                Expected Output: start_time: 01:32, end_time: 01:32
                - Query: "Show the scene at the half-hour mark."
                Expected Output: start_time: 30:00, end_time: 30:00
                - Query: "Describe the action from the beginning to 2 minutes."
                Expected Output: start_time: 00:00, end_time: 02:00
                - Query: "What happens in the last 10 seconds of the video?"
                Expected Output: start_time and end_time would depend on the video's total length

                {format_instructions}
                
                % USER QUERY:
                {query}
                
                YOUR RESPONSE:
                """
            )
            prompt = PromptTemplate(
                input_variables=["query"],
                partial_variables={"format_instructions": format_instructions},
                template=template
            )
            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            llm_result = llm_chain.run(query=query)
            final_result = output_parser.parse(llm_result)
            return final_result

    def translate(self,query:str):
        
        
        response_schemas =[
        ResponseSchema(name="translate", description="translate user's query into english")
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        format_instructions = output_parser.get_format_instructions()
        
        template = (
            """
            You are a translation expert, proficient in various languages. \n
            
            You need translate other language into English.

            {format_instructions}
            
            % USER QUERY:
            {query}
            
            YOUR RESPONSE:

            """
        )

        prompt = PromptTemplate(
            input_variables=["query"],
            partial_variables={"format_instructions": format_instructions},
            template=template 
            
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        result = output_parser.parse(llm_chain.run(query = query))
        
        return result
            