import torch.nn as nn 
import torch 


class Token(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        output  = self.emb(x)
        return output 

class Segment(nn.Module):
    def __init__(self,n_segment,d_model):
        super().__init__()
        self.seg = nn.Embedding(n_segment,d_model,padding_idx=0)
    def forward(self,x):
        output = self.seg(x)
        return output


class Position(nn.Module):
    def __init__(self,max_len,d_model):
        super().__init__()

        pe = torch.zeros(max_len,d_model).float()
        pe.zeros_grad = False 

        pos = torch.arange(0,max_len).float().unsqueeze(1)
        dev = 1/torch.pow(10000,torch.arange(0,d_model,2).float()/d_model)
        pe[:,0::2] = torch.sin(pos * dev)
        pe[:,1::2] = torch.cos(pos * dev)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        return self.pe[:,:x.size(1)]

class BertEmb(nn.Module):
    def __init__(self,vocab_size,n_segment,d_model,max_len,dropout = 0.1):
        super().__init__()
        self.pe = Position(max_len=max_len,d_model = d_model) 
        self.segment = Segment(n_segment=n_segment,d_model=d_model) 
        self.token = Token(vocab_size=vocab_size,d_model = d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        pe = self.pe(x)
        seg = self.segment(x) 
        token = self.token(x) 

        output = pe + seg + token 
        return self.dropout(output)
    