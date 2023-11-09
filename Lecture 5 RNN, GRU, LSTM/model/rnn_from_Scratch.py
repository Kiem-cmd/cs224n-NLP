import numpy as np
# from lincoln.utils.np_utils import assert_same_shape 

class RNN_Node:
    def __init__(self):
        pass 

    def forward(self, 
                x,
                h,
                params_dict): 
        """

        Args: 
            x: ndarray - shape = (B,input_dim)
                word - embeding out 
            h: ndarray - shape = (B,hidden_dim)
                hidden state
            params_dict: Dict 
                Whh - shape = (hidden_dim,hidden_dim)
                Whx - shape = (input_dim,hidden_dim)
                Ws  - shape = (hidden_dim,output_dim)
                bx  - shape = (B,hidden_dim)
                by  - shape = (B,hidden_dim)
        Return: 
            out: ndarray - shape(B,output_dim)
                output of rnn node
            h_t: ndarray - shape(B,hidden_dim)
                hidden state new

        """ 
        self.x = x 
        self.h = h 

        self.z = np.dot(self.h,params_dict['Whh']['value']) + np.dot(self.x,params_dict['Whx']['value']) + params_dict['bx']['value']
        self.h_t = np.tanh(self.z)
        self.out = np.dot(self.h_t,params_dict['Ws']['value']) + params_dict['by']['value'] 
        
        return self.out,self.h_t 
         
    def backward(self,
                d_out,
                d_h,
                params_dict): 
        """

        Args: 
            d_out: ndarray - shape = (B,output_dim)
                gradient of self.out
            d_h: ndarray - shape = (B,hidden_dim)
                gradient of self.h_t
            params_dict: Dict 
                W_hh - shape = (hidden_dim,hidden_dim)
                W_hx - shape = (input_dim,hidden_dim)
                Ws   - shape = (hidden_dim,output_dim)
                bx   - shape = (B,hidden_dim)
                by   - shape = (B,output_dim)
        Return: 
            out: ndarray - shape(B,output_dim)
                output of rnn node
            h_t: ndarray - shape(B,hidden_dim)
                hidden state new

        """ 
        params_dict['by']['grad'] += d_out.sum(axis = 0)
        params_dict['Ws']['grad'] += np.dot(self.h_t.T,d_out) 
        
        d_ht = np.dot(d_out,params_dict['Ws']['value'].T)    
        d_ht += d_h
        dz = d_ht * (1-np.tanh(self.z)*np.tanh(self.z))    

        params_dict['bx']['grad'] +=  dz.sum(axis = 0)
        params_dict['Whh']['grad'] += np.dot(self.h.T,dz) 
        params_dict['Whx']['grad'] += np.dot(self.x.T,dz)         

        dx = np.dot(dz,params_dict['Whx']['value'].T)
        dh = np.dot(dz,params_dict['Whh']['value'].T)
        

        return dx,h,params_dict

class Rnn_Layer:
    def __init__(self, 
                hidden_dim,
                output_dim,
                weight_scale = None): 
        """ 
        Args: 
        hidden_dim: int 

        output_dim: int 

        """ 

        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        self.weight_scale = weight_scale
        self.h0 = np.zeros((1,hidden_dim))
        self.first = True 
    def init(self, 
            input_):
        
        """
        Args: 
            input_ : ndarray 
                shape(input_) = (B,Seq_len,Vocab_size)
        """
        self.input_dim = input_.shape[2]
        if not self.weight_scale: 
            self.weight_scale = 2/(self.input_dim + self.output_dim)

        self.params = {}
        self.params['Ws'] = {} 
        self.params['Whh'] = {} 
        self.params['Whx'] = {} 
        self.params['bx'] = {} 
        self.params['by'] = {} 


        self.params['Ws']['value'] = np.random.normal(loc = 0.0,
                                                      scale = self.weight_scale,
                                                      size = (self.hidden_dim,self.output_dim)) 
        self.params['Whh']['value'] = np.random.normal(loc = 0.0,
                                                     scale = self.weight_scale,
                                                     size = (self.hidden_dim,self.hidden_dim))

        self.params['Whx']['value'] = np.random.normal(loc = 0.0,
                                                     scale = self.weight_scale,
                                                     size = (self.input_dim,self.hidden_dim)) 
        self.params['bx']['value'] = np.random.normal(loc = 0.0,
                                                     scale = self.weight_scale,
                                                     size = (1,self.hidden_dim)) 
        self.params['by']['value'] = np.random.normal(loc = 0.0,
                                                    scale = self.weight_scale,
                                                    size = (1,self.output_dim)) 

        self.params['Ws']['grad'] = np.zeros_like(self.params['Ws']['value'])
        self.params['Whh']['grad'] = np.zeros_like(self.params['Whh']['value'])
        self.params['Whx']['grad'] = np.zeros_like(self.params['Whx']['value'])
        self.params['bx']['grad'] = np.zeros_like(self.params['bx']['value'])
        self.params['by']['grad'] = np.zeros_like(self.params['by']['value'])

        self.cells = [RNN_Node() for x in range(input_.shape[1])] 

    def clear_gradient(self):
        for i in self.params.keys():
            self.params[i]['grad'] = np.zeros_like(self.params[i]['value'])

    def forward(self,
                x_seq):
        """
        Args: 
            x_seq: ndarray 
            ..................
        Return: 

        """ 
        if self.first: 
            self.init(x_seq) 
            self.first  = False  
        batch_size, seq_len, input_dim = x_seq.shape         
        h_in = np.copy(self.h0) 
        h_in = np.repeat(h_in,batch_size,axis = 0)
        x_out = np.zeros((batch_size,seq_len,self.output_dim))
        for t in range(seq_len):
            x_in = x_seq[:,t,:]
            out,h = self.cells[t].forward(x_in,h_in,self.params) 
            x_out[:,t,:] = out 

        self.h0 = h_in.mean(axis = 0, keepdims = True)

        return x_out
    
    def backward(self,
                x_seq_out_grad): 
        """ 
        Args: 
        x_out_grad :  array
            shape - (B,seq_len,output_dim)
        """ 

        batch_size,seq_len,output_dim = x_seq_out_grad.shape 
        h_in_grad = np.zeros((batch_size,self.hidden_dim))
        x_seq_in_grad = np.zeros((batch_size,seq_len,self.input_dim)) 

        for t in reversed(range(seq_len)):
            x_out_grad = x_seq_out_grad[:,t,:] 
            grad_out, h_in_grad = self.cells[t].backward(x_out_grad,h_in_grad,self.params) 
            x_seq_in_grad[:,t,:] = grad_out 
        return x_seq_in_grad


class RNNModel():
    def __init__(self,
                layer,
                sequence_length,
                vocab_size,
                loss): 
        """ 
        Args: 
        ---------------------------
        layer: RNN_Layer
        ................
        sequen_lenth: int 
        .................
        vocab_size: int 
        .................
        loss : Loss
        .................

        """ 

        self.layers = layers 
        self.vocab_size = vocab_size 
        self.sequence_length = sequence_length 
        self.loss = loss 
        for layer in self.layers:
            pass 

    def forward(self,x_batch):
        """


        """

        for layer in self.layers:
            x_batch = layer.forward(x_batch) 
        return x_batch
    def backward(self,
                 loss_grad):

        """  

        """
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
        return loss_grad
