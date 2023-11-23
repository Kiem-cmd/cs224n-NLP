import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from char_emb import CharEmb 
from word_emb import WordEmb 
from highway_network import HighWay
from contextualLayer import Contextual 
from attentionFlow import attentionFlow
from modelingLayer import modellingLayer 
from outputLayer import outputLayer 
import argparse

class BiDAF(nn.Module):
    def __init__(self,args):
        super().__init__() 
        ## 1. Char embedding 
        self.char_emb1 = CharEmb(args)
        self.char_emb2 = CharEmb(args)

        ## 2. Word embedding 
        self.word_emb1 = WordEmb(args)
        self.word_emb2 = WordEmb(args)

        ## 3. Highway Network
        self.high_way1 = HighWay(args) 
        self.high_way2 = HighWay(args) 

        ## 4. Contextual embedding layer
        self.context_layer1 = Contextual(args)
        self.context_layer2 = Contextual(args)
        ## 5. Attention Flow layer 
        self.attn_flow_layer = attentionFlow(args)
        ## 6. Modeling layer
        self.modeling_layer = modellingLayer(args)
        ## 7. Output layer
        self.output_layer = outputLayer(args)

    def forward(self,context_char,context_word,query_char,query_word): 
        ### Stage 1: char_emb & word_emb 
        
        # context_char = batch.context_char   ##  
        # context_word = batch.context_word   ## 
        # query_char = batch.query_char       ##
        # query_word = batch.query_word       ##   
        context_char = self.char_emb1(context_char) 
        context_word = self.word_emb1(context_word) 

        query_char = self.char_emb2(query_char)
        query_word = self.word_emb2(query_word)

        ### Stage 2: highway network 
        context_emb = torch.cat([context_char,context_word],-1)
        query_emb = torch.cat([query_char,query_word],-1)
        context_emb = self.high_way1(context_emb) 
        query_emb = self.high_way2(query_emb)

        ### Stage 3: Build contextual emb
        context_emb = self.context_layer1(context_emb)
        query_emb = self.context_layer2(query_emb)
        
        ### Stage 4: Calculate S & G 
        
        g = self.attn_flow_layer(context_emb,query_emb)
  
        ### Stage 5: Modeling layer  
        m = self.modeling_layer(g)
        ### Stage 6: Output layer 
        p1,p2 = self.output_layer(g,m)
        
        return p1,p2 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--batch-size',default = 10) 
    parser.add_argument('--hidden-size',default = 400) 
    parser.add_argument('--seq-len',default=40)
    parser.add_argument('-char-vocab-size',default = 300) 
    parser.add_argument('--char-dim',default = 100)
    parser.add_argument('--char-channel-size',default = 100) 
    parser.add_argument('--char-width',default = 5) 
    parser.add_argument('--word-vocab-size',default=1000)
    parser.add_argument('--word-dim',default = 300)
    parser.add_argument('--num-layers',default = 3) 
    args = parser.parse_args() 

    model = BiDAF(args)
    batch_sample = [] 
    context_char = torch.randint(200,(10,40,10))  ## B,seq_len,word_len
    context_word = torch.randint(200,(10,40))            ## B,seq_len
    query_char = torch.randint(200,(10,20,10))            ## B,seq_len,word_len
    query_word = torch.randint(200,(10,20))                 ## B,seq_len

    p1,p2 = model(context_char,context_word,query_char,query_word)
    print(p1.shape == (args.batch_size,args.seq_len)) 
    print(p2.shape == (args.batch_size,args.seq_len))