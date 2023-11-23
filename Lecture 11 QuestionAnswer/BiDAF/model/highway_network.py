import torch 
import torch.nn as nn
import torch.nn.functional as F 
import argparse

class HighWay(nn.Module):
    def __init__(self,args,act_func = nn.ReLU()):
        super().__init__()
        self.input_dim = args.hidden_size
        self.num_layers = args.num_layers
        self.act_func = act_func
        
        self.linear = nn.ModuleList([
            nn.Linear(self.input_dim,self.input_dim) 
            for _ in range(self.num_layers)
        ]) 
        self.gate = nn.ModuleList([
            nn.Linear(self.input_dim,self.input_dim) 
            for _ in range(self.num_layers)
        ])
 
    def forward(self,x):
        assert x.size(-1) == self.input_dim, "word_dim + char_channel_size ko =  hidden size"
        for layer in range(self.num_layers):
            linear = self.linear[layer](x)
            linear = self.act_func(linear)
            gate = self.gate[layer](x)
            gate = torch.sigmoid(gate)
            
            x = gate * linear + (1-gate) * x
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--hidden-size',default = 20 ,type = int)
    parser.add_argument('--num_layers',default = 2,type = int ) 
    args = parser.parse_args() 

    x = torch.rand(2,10,20)
    high_way = HighWay(args) 
    print(high_way(x).shape)






