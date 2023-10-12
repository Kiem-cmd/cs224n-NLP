import torch 
import torch.nn as nn
 
class FeedForward(nn.Module):
    def __init__(self,d_model,ff_dim, drop = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model,ff_dim) 
        self.relu = nn.ReLU() 
        self.linear2 = nn.Linear(ff_dim,d_model)
        self.dropout = nn.Dropout(drop)
    def forward(self,x): 
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

if __name__ == '__main__':
    x = torch.rand(10,20,56) 
    ffn =FeedForward(56,56*4)
    print(ffn(x).size())