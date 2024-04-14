'''
Author: VideoPal Team
Date: 2024-03-22 21:43:17
LastEditors: VideoPal Team
LastEditTime: 2024-03-24 23:05:31
FilePath: /chengruilai/projects/VideoPal/utils/inputparser.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import langid

from models.translate_model import Translator

class TimeParser:
    @staticmethod
    def time_to_seconds(time_str: str) -> int:
        """Converts time string to seconds."""
        parts = time_str.split(':')
        parts_count = len(parts)
        if parts_count == 3:  # HH:MM:SS format
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif parts_count == 2:  # MM:SS format
            return int(parts[0]) * 60 + int(parts[1])
        elif parts_count == 1:  # SS format
            return int(parts[0])
        else:
            raise ValueError("Invalid time format")
    
    @staticmethod
    def convert_time_format(seconds: int) -> str:
        """Converts seconds to formatted time."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h{minutes}min{seconds}s"
    
def translate_zh_en(query: str, translator: Translator):
        langid.set_languages(["en", "zh"])
        lid = langid.classify(query)
        if lid[0] == "en":
            return query
        else:
            return translator(query)