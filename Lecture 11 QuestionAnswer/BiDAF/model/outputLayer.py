import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import argparse 


class outputLayer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.batch_size = args.batch_size 
        self.hidden_size = args.hidden_size


        self.w_p1 = nn.Linear(5 * self.hidden_size,1)
        self.w_p2 = nn.Linear(5 * self.hidden_size,1)

        self.lstm = nn.LSTM(int(self.hidden_size),int(self.hidden_size/2),bidirectional=True,batch_first=True)
    def forward(self,g,m):
        concat = torch.cat((g,m),2) 
        p_1 = self.w_p1(concat).squeeze() 
        h0 = torch.zeros(2*1,self.batch_size,int(self.hidden_size/2)).normal_(0.0,0.01)
        c0 = torch.zeros(2*1,self.batch_size,int(self.hidden_size/2)).normal_(0.0,0.01) 
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        h0 = nn.Parameter(h0,requires_grad=True)
        c0 = nn.Parameter(c0,requires_grad=True)
        m_2,_ = self.lstm(m,(h0,c0))
        concat2 = torch.cat((g,m_2),2) 
        p_2 = self.w_p2(concat2).squeeze()
        return F.softmax(p_1,dim =1),F.softmax(p_2,dim = 1) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size',default=10)
    parser.add_argument('--hidden-size',default=50) 
    args = parser.parse_args()
    g = torch.rand(args.batch_size,100,4 * args.hidden_size)
    m = torch.rand(args.batch_size,100,args.hidden_size)
    layer = outputLayer(args)
    p1,p2 = layer(g,m) 
    print(F.softmax(p1,dim = 1).shape)
    print(p2.shape)