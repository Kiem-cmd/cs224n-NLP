import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import argparse 


class modellingLayer(nn.Module):
    def __init__(self,args,drop_rate = 0.2):
        super().__init__() 
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size 
        self.layer = nn.LSTM(int(4*self.hidden_size),int(self.hidden_size/2),
                            num_layers=2,
                            bidirectional=True, 
                            batch_first = True,
                            dropout=drop_rate) 

    def forward(self,g):
        h0 = torch.zeros(4,self.batch_size,int(self.hidden_size/2)).normal_(0.0,0.02) 
        c0 = torch.zeros(4,self.batch_size,int(self.hidden_size/2)).normal_(0.0,0.02) 
        if torch.cuda.is_available():
            h0 = h0.cuda() 
            c0 = co.cuda() 
        h0 = nn.Parameter(h0,requires_grad=True)
        c0 = nn.Parameter(c0,requires_grad=True) 

        m,_ = self.layer(g,(h0,c0)) 
        return m

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--batch-size',default=10,type=int)
    parser.add_argument('--hidden-size',default=50,type=int)
    args = parser.parse_args() 
    g = torch.rand(10,100,200) 
    layer = modellingLayer(args)
    m = layer(g)
    if (m.shape == (args.batch_size,g.shape[1],args.hidden_size)):
        print("Pass")
    else:
        "xxxxxxxxxxxxxxxxxxxxxxxxxxx"