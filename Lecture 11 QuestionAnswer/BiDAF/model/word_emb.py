import torch
import torch.nn as nn 
import torch.nn.functional as F 
import argparse 
class WordEmb(nn.Module):
    def __init__(self,args,is_pretrain = False):
        super().__init__()
        if is_pretrain: 
            self.emb = nn.Embedding.from_pretrained(pretrain,freeze = True)
        else:
            self.emb = nn.Embedding(args.vocab_word_size,args.word_dim)
        
    def forward(self,x):
        '''
        Params:
            x: (batch_size, sentence_len)
        
        Return: 
            x: (batch_size, sentence_len, word_dim) 

        '''
        x = self.emb(x)
        return x

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--vocab_word_size',default=300,type =int)
    parser.add_argument('--word_dim',default = 10,type = int) 
    args = parser.parse_args()
    x = torch.randint(0,200,(10,10))
    word_emb = WordEmb(args)
    print(word_emb(x).shape) 

if __name__ == '__main__':
    main()