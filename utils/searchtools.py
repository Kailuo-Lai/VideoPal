'''
Author: VideoPal Team
Date: 2024-03-22 21:46:41
LastEditors: VideoPal Team
LastEditTime: 2024-03-24 23:05:11
FilePath: /chengruilai/projects/VideoPal/utils/searchtools.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import numpy as np
from typing import List

def cosine_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
    - embedding1 (array): A 1-D array representing the first embedding.
    - embedding2 (array): A 1-D array representing the second embedding.
    
    Returns:
    - float: The cosine similarity between the two embeddings.
    """
    # Normalize the embeddings to have unit length
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Compute the dot product between the normalized embeddings
    similarity = np.dot(embedding1_norm, embedding2_norm)
    
    return similarity
        
        
def get_clip_embedding(video_info: List):
    '''
    video_info: [vid, video_name, video_length, filepath]
    '''
    data = np.load(f"./tmp/{video_info[0]}_{video_info[1]}/clip_embed.npz")
    return data['features']
    
    
