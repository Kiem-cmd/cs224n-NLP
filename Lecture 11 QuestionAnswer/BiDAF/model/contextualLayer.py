import torch 
import torch.nn as nn 

class Contextual(nn.Module):
    def __init__(self,args):
        super().__init__() 
        self.context_layer = nn.LSTM(input_size = int(args.hidden_size),
                                    hidden_size = int(args.hidden_size/2),
                                    bidirectional=True,
                                    batch_first = True,
) 

    def forward(self,x):
        out, _ = self.context_layer(x)
        return out