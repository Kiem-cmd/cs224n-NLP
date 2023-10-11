import torch.nn as nn 
import torch 

class MultiHeadAttention(nn.Module):
    def __init__(self,num_head,d_model,dropout = 0.1):
        super().__init__() 
        
        assert d_model % num_head == 0, "d_model phai chia het cho num_head"
        d_k = d_model // num_head

        self.W_q = nn.Linear(d_model,d_k)
        self.W_k = nn.Linear(d_model,d_k)
        self.W_v = nn.Linear()

        self.W_o = nn.Linear(d_model,)

        self.dropout = nn.Dropout(dropout)

    def Scale_Dot_Product_Attention(self,q,k,v,mask = None):
        score = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(d_model)
        return output 


    def forward(self,x,mask):
        q = self.W_q(q) 
        k = self.W_k(k)
        v = self.W_v(v) 

        q = q.view(q.shape[0],q.shape[1],self.num_head,self.d_k)
        k = k.view(k.shape[0],k.shape[1],self.num_head,self.d_k) 
        v = v.view(v.shape[0],v.shape[1],self.num_head,self.d_k) 

        x,attn_score = self.Scale_Dot_Product_Attention(q,k,v,mask) 
        
        x = self.dropout(x) 
        x.transpose(1,2).contiguous().view(x.shape[0],-1,self.n_heads * self.d_k)
        
        return self.W_o(x),attn_score 

        
