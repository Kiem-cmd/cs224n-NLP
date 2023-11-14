import torch
import torch.nn as nn 
import torch.nn.function as F 

class WordEmb(nn.Module):
    def __init__(self,args,is_pretrain = False):
        super().__init__()
        if is_pretrain: 
            self.emb = nn.Embedding.from_pretrained(pretrain,freeze = True)
        else:
            self.emb = nn.Embedding(args.vocab_word_size,args.word_dim)
        
    def forward(self,x):
        x = self.emb(x)
        return x