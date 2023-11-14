import torch 
import torch.nn
import torch.nn.function as F 

class HighWay(nn.Module);
    def __init__(self,args,act_func = nn.ReLU()):
        super().__init__()
        self.input_dim = args.hidden_size
        self.num_layer = args.num_layer
        
        self.linear = nn.ModuleList([
            nn.Linear(input_dim,input_dim) 
            for _ in range(num_layers)
        ]) 
        self.gate = nn.ModuleList([
            nn.Linear(input_dim,input_dim) 
            for _ in range(num_layers)
        ])
 
    def forward(self,x):
        for layer in range(self.num_layers):
            linear = self.linear[layer](x)
            linear = self.act_func(linear)
            gate = self.gate[layer](x)
            gate = torch.sigmoid(gate)
            
            x = gate * linear + (1-gate) * x
        return x