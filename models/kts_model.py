'''
Author: VideoPal Team
Date: 2024-03-02 11:20:30
LastEditors: VideoPal Team
LastEditTime: 2024-04-14 12:33:53
FilePath: /root/autodl-fs/projects/VideoPal/models/kts_model.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import numpy as np
from models.kts_src.kts_utils import cpd_auto, l2_normalize_np_array

class KTSModel:
    def __init__(self, args):
        self.alpha = args.alpha
        self.vmax = args.vmax
        
    def __call__(self, video_features, video_length, inter_seconds: int = 1):
        '''
        Get segment windows from video using KTS algorithm.
        video_features: CLIP or other embeddings.
        video_length: video length in seconds.
        inter_seconds: The smallest time gap between successive clips, in seconds.
        '''
        video_features = video_features[::inter_seconds]
        K = l2_normalize_np_array(video_features)
        K = np.dot(K, K.T)
        clip_num = K.shape[0]
        max_seg_num = clip_num // self.alpha
        
        cps, _ = cpd_auto(K, max_seg_num - 1, vmax=self.vmax)
        seg_num = len(cps) + 1

        seg_points = [x * inter_seconds for x in cps]
        seg_points = np.insert(seg_points, 0, 0)
        seg_points = np.append(seg_points, video_length)
    
        seg_windows = [(int(seg_points[i]), int(seg_points[i+1])) for i in range(seg_num)]
        return seg_windows
