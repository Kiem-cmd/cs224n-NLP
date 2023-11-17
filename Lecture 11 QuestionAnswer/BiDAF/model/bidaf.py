import torch 
import torch.nn as nn 
import torch.nn.funtional as F 
from char_embed import CharEmb 
from word_embed import WordEmb 
from high_way import HighWay


class BiDAF(nn.Module):
    def __init__(self,args):
        super().__init__() 
        ## 1. Char embedding 
        self.char_emb = CharEmb(args)
        ## 2. Word embedding 
        self.word_emb = WordEmb(args)
        ## 3. Highway Network
        self.high_way = HighWay(args) 
        
        ## 4. Contextual embedding layer
        self.context_layer = nn.LSTM(input_size = args.hidden_size,
                                    hidden_size = args.hidden_size/2,
                                    bidirection = True,
                                    batch_first = True,
                                    drop_rate = args.drop_rate
                                    ) 
        ## 5. Attention Flow layer 
        self.attn_weight_context = nn.Linear(args.hidden_size * 2, 1) 
        self.attn_weight_query = nn.Linear(args.hidden_size * 2, 1) 
        self.attn_weight_cq = nn.Linear(args.hidden_size * 2, 1) 
        self.attn_weight_qc = nn.Linear(args.hidden_size * 2, 1) 
        ## 6. Modeling layer
        self.modeling_layer = nn.LSTM() 
        self.modeling_layer2 = nn.LSTM() 
        ## 7. Output layer
        
    def build_contextual_emb(self,emb):
        out, _ = self.contextual_emb_layer(emb)
        return out
    def attn_flow_layer(self,c,q):
        """
        Params: 
        
            H: (batch,context_len, hidden_size * 2)
            U: (batch, query_len, hidden_size * 2) 
        
        Return: 
            G: (batch,context_len, query_len)
        """
        
        ## S = w(s)[H,U,HoU]
        HoU = []
        for i in range(q.size(1)):
            qi = 
            ci = 
            cq.append(ci)
        HoU = torch.stack(HoU, dim = -1) 
        S = torch.cat()
        S = self.attn_weight_context(H) + self.attn_weight_query(U) + HoU
        U_hat = torch.bmm(a,q)
        
        H_hat = 
        ### G = [H;U_hat;HoU_hat;HoH_hat] 
        G = torch.cat([H,U_hat,H*U_hat,H*H_hat], dim = -1) 
        return G
    def modeling_layer(self,g):
        return m
    def output_layer(self,g,m):
        return p1,p2
    def forward(self,batch): 
        ### Stage 1: char_emb & word_emb 
        
        context_char = batch.context_char   ##  
        context_word = batch.context_word   ## 
        
        query_char = batch.query_char       ##
        query_word = batch.query_word       ##   
        
        ### Stage 2: highway network 
        context_emb = torch.cat((context_char,context_word),2)
        query_emb = torch.cat((query_char,query_word),2)
        
        context_emb = self.highway(context_emb) 
        query_emb = self.highway(query_emb)
        ### Stage 3: Build contextual emb

        context_emb = build_contextual_emb(context_emb)
        query_emb = build_contextual_emb(query_emb)
        
        ### Stage 4: Calculate S & G 
        
        shape = (batch_size,T,J,2*self.d)
        
        context_emb_ex = context_emb.unsqueeze(2)
        context_emb_ex = context_emb.ex.expand(shape)
        
        query_emb_ex = context_emb.unsqueeze(2)
        query_emb_ex = context_emb_ex.expand(shape) 
       
        elwise_mul = torch.mul((context_emb_ex,query_emb_ex),3)
        concate = torch.cat((context_emb_ex,query_emb_ex,elwise_mul),3) 
        S = self.W(concate).squeeze(3)
        
        
        ## Context -> Query
        c2q = torch.bmm(F.softmax(S,dim= - 1), query_emb)  
        
        ## Query -> Context
        b = F.softmax(torch.max(S,2)[0],dim  = -1))
        q2c = torch.bmm(b.unsqueeze(1),context_emb).squeeze()
        q2c = q2c.unsqueeze(1).expand(-1, T, -1)
        ## query-aware representation for each context word
        G = torch.cat([context_emb,c2q,c * c2q,c*q2c], dim = -1)
  
        ### Stage 5: Modeling layer  
        
        ### Stage 6: Output layer 
        p1,p2 = output_layer (g,m,c_lens)
        
        return p1,p2 
        
        
def main():
    
    
    
if __name__ == '__main__':
    main()
        