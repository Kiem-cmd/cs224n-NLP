import numpy as np 


class Rnn:
    def __init__(self,input_dim,hidden_dim,ouput_dim,num_layer,bidirectional):
        self.input_dim = input_dim  
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        self.num_layer = num_layer 
        self.bidirectional = bidirectional 

    def init(self):
        self.W_hh = np.random.rand(self.hidden_dim,self.hiddend_size) 
        self.W_hx = np.random.rand(self.hidden_dim,self.input_dim)
        self.W_s = np.random.rand(self.output_size,self.hidden_size)

    def forward(self,inputs):
        list_hidden_state = [] 
        list_out = []
        h = np.zeros((self.hidden_size,1))
        list_hidden_state.append(h)

        for i in range(len(inputs)):
            h_t = np.tanh(self.W_hh * list_hidden_state[i] + self.W_hx * inputs[i] + self.bias) 
            list_hidden_state.append(h_t)
            out = np.softmax(self.W_s * h_t + self.by)
            list_out.append(out)
        return y,h
    def loss(self,d_y, learning_rate = 0.01):
        d_Ws = d_y * self.las
         
    def backward(self):
        pass 
    def update(self):
        pass 
