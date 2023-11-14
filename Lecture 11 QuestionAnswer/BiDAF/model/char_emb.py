import torch 
import torch.nn as nn
import torch.nn.function as F 

class CharEmb(nn.Module):
    def __init__(self,args,drop_rate = 0.2):
        super().__init__()
        args = self.args 
        
        self.dropout = nn.Dropout(drop_rate)
        self.emb = nn.Embedding(args.char_vocab_size,args.char_dim)
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels = 1,
                                  out_channels = args.char_channel_size,
                                  kernel_size = (args.char_dim,args.char_width)),
                        nn.ReLU()
                        )
    def forward(self,x):
        ## x - (B,seq_len,word_len)
        batch_size,seq_len,word_len = x.size()
        x = x.view(-1,word_len) ### x.shape = (B.seq_len,word_len) 
        x = self.dropout(self.emb(x))       ### x.shape = (B.seq_len,word_len,char_dim)
        x = x.transpose(1,2).unsqueeze(1) ## x.shape = (B.seq,1,char_dim,word_len)  <=>  (batch_size,channel,height,width) in CV 
        x = self.conv(x).squeeze()     ##   x.shape = (B.seq,char_channel_size,1,W_out) tai kernel size = height nen shape[2] = 1
        
        x = F.maxpool1d(x,x.size(2)).squeeze() ## x.shape = (B.seq,char_channel_size)
        x = x.view(batch_size,-1,self.args.char_channel_size) 
        
        return x