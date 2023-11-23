import torch 
import torch.nn as nn 

class Contextual(nn.Module):
    def __init__(self):
        super().__init__(args,drop_rate = 0.2) 
        self.context_layer = nn.LSTM(input_size = args.hidden_size,
                                    hidden_size = args.hidden_size/2,
                                    bidirectional=True,
                                    batch_first = True,
                                    drop_rate = drop_rate,) 

    def forward(self,x):
        out, _ = self.context_layer(x)
        return out