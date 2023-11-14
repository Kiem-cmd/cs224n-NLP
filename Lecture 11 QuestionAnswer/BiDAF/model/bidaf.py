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
        self.context_layer = nn.LSTM(input_size = args.hidden_size * 2,
                                    hidden_size = args.hidden_size,
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
        
    def build_contextual_emb(self,concate_char_word):
        contextual_embed, _h = self.context_layer(concate_char_word) 
        return contextual_embed
    def attn_flow_layer(self,H,U):
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
        ### Stage 1: 
        context_char = batch.context_char 
        query_char = batch.query_char 
        context_word = batch.context_word
        query_word = batch.query_word 
        ### Stage 2: 
        context_emb = build_contextual_emb(context_char,context_word)
        query_emb = build_contextual_emb(query_char,query_word)
        ### Stage 3: 
        
        context_emb = self.context
        
        similar = torch.mul(context_emb,query_emb) 
        cat_data = torch.cat((context_emb,query_emb,similar),3)
        
        S = self.W(cat_data).view()
        ## Context -> Query
        c2q = torch. 
        ## Query -> Context
        q2c = torch. 
        ## query-aware representation for each context word
        G = torch.cat((context))
        
        ### Stage 4: 
        
        p1,p2 = output_layer (g,m,c_lens)
        
        return p1,p2 
        
        

        