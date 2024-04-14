'''
Author: VideoPal Team
Date: 2024-03-23 22:20:19
LastEditors: VideoPal Team
LastEditTime: 2024-03-24 23:06:32
FilePath: /chengruilai/projects/VideoPal/models/text_embedding_model.py
Description: Pal for Long Video Chat
Copyright (c) 2024 by VideoPal Team, All Rights Reserved. 
'''

from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer,logging
from ipex_llm.optimize import optimize_model
from utils.utils import new_cd
import torch.nn.functional as F
import os
logging.set_verbosity_error()

parent_dir = os.path.dirname(__file__)

class TextEmbedding:
    def __init__(self, args):
        with new_cd(parent_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(f"../checkpoints/{args.embed_version}",trust_remote_code=True)
            self.model = AutoModel.from_pretrained(f"../checkpoints/{args.embed_version}",trust_remote_code=True)
            if args.embed_low_bit:
                self.model = optimize_model(self.model,low_bit='sym_int4')
            
    def average_pool(self,last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def __call__(self, texts):
        batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings