'''
Author: VideoPal Team
Date: 2024-03-02 11:20:30
LastEditors: VideoPal Team
LastEditTime: 2024-03-12 11:12:46
FilePath: /chengruilai/projects/VideoPal/models/tag2text_model.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

import os
import torchvision.transforms as transforms

from models.tag2text_src.tag2text import tag2text_caption
from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)

class Tag2TextModel:
    def __init__(self, args):
        self.threshold = args.tag2text_thershld # threshold for tagging, default 0.68
        self.init_model()
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
    
    def init_model(self):
        # delete some tags that may disturb captioning
        # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
        delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]

        #######load model
        with new_cd(parent_dir):
            model = tag2text_caption(pretrained='../checkpoints/tag2text_swin_14m.pth',
                                    image_size=384,
                                    vit='swin_b',
                                    delete_tag_index=delete_tag_index,
                                    threshold = self.threshold)
        model.eval()
        # model = optimize_model(model)
        self.model = model