import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
class CharEmb(nn.Module):
    def __init__(self,args,drop_rate = 0.2):
        super().__init__()
        self.args = args 
        
        self.dropout = nn.Dropout(drop_rate)
        self.emb = nn.Embedding(args.char_vocab_size,args.char_dim)
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels = 1,
                                  out_channels = args.char_channel_size,
                                  kernel_size = (args.char_dim,args.char_width)),
                        nn.ReLU()
                        )
    def forward(self,x):
        """ 
        Params: 
            x - (B,seq_len,word_len)
        Return: 
            x - (B,seq_len,char_channel_size)
        """ 
        batch_size,seq_len,word_len = x.size()
        x = x.view(-1,word_len) ### x.shape = (B.seq_len,word_len) 
        x = self.dropout(self.emb(x))       ### x.shape = (B.seq_len,word_len,char_dim)
        x = x.transpose(1,2).unsqueeze(1) ## x.shape = (B.seq,1,char_dim,word_len)  <=>  (batch_size,channel,height,width) in CV 
        x = self.conv(x).squeeze()     ##   x.shape = (B.seq,char_channel_size,1,W_out) tai kernel size = height nen shape[2] = 1
        
        x = F.max_pool1d(x,x.size(2)).squeeze() ## x.shape = (B.seq,char_channel_size)
        x = x.view(batch_size,-1,self.args.char_channel_size) 
        
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--char_vocab_size',default= 300,type = int)
    parser.add_argument('--char_dim',default = 20,type = int)
    parser.add_argument('--char_channel_size', default=10, type = int )
    parser.add_argument('--char_width',default=3,type = int) 
    args = parser.parse_args()
    char_emb = CharEmb(args,drop_rate= 0.3)
    x = torch.randint(20,200,(10,10,10))
    print(char_emb(x).shape)
