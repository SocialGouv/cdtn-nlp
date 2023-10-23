from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from cleantext import clean
from torch import Tensor
from pandas import Series
import pandas as pd
import math


class Embeddings :

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

    def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def make_windows(content, w_size):
        tokens = content.split(" ")
        l = len(tokens)    
        
        n = math.ceil(l / w_size)        

        if n==1 :
            return [content]
        else :
            windows = []
            for i in range(n) :
                x = i*w_size
                y = (i+1)*w_size
                w = tokens[x:y]
                windows.append(" ".join(w))
    
        return windows

    def preprocessText(text, isPassage: bool):
        prefix = 'passage: ' if isPassage else 'query:'
        return prefix + clean(str(text), to_ascii=False)

    def runEmbed(self, p) :
        batch_dict = self.tokenizer(p, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1).detach()
        return normalized_embeddings

    def embedQP(self, questions: Series, passages: Series):
        # add prefix and clean
        pp_questions = questions.apply(lambda q : self.preprocessText(q, False))
        pp_passages = passages.apply(lambda p : self.preprocessText(p, True))
        pp = pd.concat([pp_passages, pp_questions], axis=0).to_numpy().tolist()
        return self.runEmbed(pp).numpy()