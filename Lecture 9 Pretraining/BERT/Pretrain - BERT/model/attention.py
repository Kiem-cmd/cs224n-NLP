import torch.nn as nn 
import torch 

class Scale_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self,q,k,v,mask = None): 
        d_k = torch.tensor(q.shape[-1])
        qk_T = torch.matmul(q,k.transpose(-1,-2)) 
        scaled = qk_T / torch.sqrt(d_k) 
        if mask is not None: 
            scaled.mask_fill(mask, 1e-14) 
        attn_score = nn.Softmax(dim = -1)(scaled)
        output = torch.matmul(attn_score,v)

        return output, attn_score
class MultiHeadAttention(nn.Module):
    def __init__(self,num_head,d_model,dropout = 0.1):
        super().__init__() 
        
        assert d_model % num_head == 0, "d_model phai chia het cho num_head"
        self.d_k = d_model // num_head

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)

        self.W_o = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout)
        self.num_head = num_head
    def forward(self,q,k,v,mask = None):
        batch_size = q.shape[0]

        q = self.W_q(q) 
        k = self.W_k(k)
        v = self.W_v(v) 

        q = q.view(q.shape[0],q.shape[1],self.num_head,self.d_k).transpose(1,2)
        k = k.view(k.shape[0],k.shape[1],self.num_head,self.d_k).transpose(1,2)
        v = v.view(v.shape[0],v.shape[1],self.num_head,self.d_k).transpose(1,2)

        x,attn_score = Scale_Dot_Product_Attention()(q,k,v,mask) 
        
        output = self.dropout(x) 
        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.num_head * self.d_k)
        
        return self.W_o(output),attn_score 

if __name__ == "__main__" :
    attn = MultiHeadAttention(8,56)

    x = torch.rand(10,20,56)        
    a,b = attn(x,x,x)
    print(a.size())