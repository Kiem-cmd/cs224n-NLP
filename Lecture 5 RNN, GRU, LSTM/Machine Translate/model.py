import torch.nn as nn 
"""
Document Pytorch:

output,h_n = nn.RNN(input,h0) 
    input: 
    - input.shape = (L,H_in)
    - h0.shape = (D * n_layers,H_out)
    output:
    -  
"""


class Encoder(nn.Module):
    def __init__(self,vocab_len,embed_dim,hidden_dim,n_layers,dropout_rate):

        self.embedding = nn.Embedding(vocab_len, embed_dim)

        self.rnn = nn.RNN(embed_dim,hidden_dim,n_layers) 
        self.gru = nn.GRU()
        self.lstm = nn.LSTM() 

        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self,x,type_):
        embed = self.dropout(self.embedding(x))
        
        if type_ == "RNN":
            outputs,(hidden,cell) = self.rnn(embed)
        elif type_ == "GRU":
            outputs,(hidden,cell) = self.gru(embed) 
        elif type_ == "LSTM":
            outputs,(hidden,cell) = self.lstm(embed) 


        assert hidden,cell,"Error" 

class Decoder(nn.Module):
    def __init__(self,output_dim,emb_dim,hiddien_dim,n_layers,dropout_rate):
        super().__init__()
        
        self.output_dim = output_dim 
        self.hidden_dim = hidden_dim 
        self.emb_dim = emb_dim 

        self.embdding = nn.Embedding(output_dim,emb_dim)
        self.rnn = nn.LSTM(emb_dim,hiddien_dim,n_layers,dropout = dropout_rate)      
        self.fc_out = nn.Linear(hiddien_dim,output_dim)
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self,x,hidden,cell):
        embedding = self.drop_out(self.embdding(x))
        output,(hidden,cell) = self.rnn(embdding,(hidden,cell))

        predict = self.fc_out(output)
        return predict,hidden,cell


class nmt(nn.Module):
    def __init__(self,encoder,decoder,device):
        super().__init__()
        
        self.encoder = encoder 
        self.decoder = decoder 
        self.device = device 

        assert encoder.hidden_dim == decoder.hidden_dim, "Hidden dimension of encoder and decoder must equal!!!"
        assert encoder.n_layers == decoder.n_layers,"Number layers of encoder and decoder must equal!!!"
    def forward(self,x,trg):
        batch_size = 
         
