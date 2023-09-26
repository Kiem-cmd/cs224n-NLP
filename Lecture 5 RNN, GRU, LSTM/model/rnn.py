import numpy as np
from lincoln.utils.np_utils import assert_same_shape 

class RNN_Node:
    def __init__(self):
        pass 

    def forward(self, 
                x: ndarray, 
                h: ndarray,
                params_dict: Dict[str,ndarray]): 
        """

        Args: 
            x: ndarray - shape = (B,embed_dim)
                word - embeding out 
            h: ndarray - shape = (B,hidden_dim)
                hidden state
            params_dict: Dict 
                W_hh,W_hx,Ws,bx,by 
        Return: 
            out: ndarray - shape(B,output_dim)
                output of rnn node
            h_t: ndarray - shape(B,hidden_dim)
                hidden state new

        """ 


        self.x = x 
        self.h = h 


        self.z = np.dot(params_dict['W_hh']['value'],self.h.T) + np.dot(params_dict['W_hx']['value'],self.x.T) + params_dict['bx']['value']
        self.h_t = np.tanh(self.z)
        self.out = np.dot(params_dict['W_s']['value'],self.h_t.T) + params_dict['by']['value'] 

        return self.out,self.h_t 
         
    def backward(self,
                d_out : ndarray,
                d_h   : ndarray,
                params_dict: Dict[str,ndarray]): 
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
        ## Check shape
        assert_same_shape(d_out,self.out)
        assert_same_shape(d_h,self.h) 


        params_dict['by']['grad'] += d_out.sum(axis = 0)
        params_dict['Ws']['grad'] += np.dot(self.h_t.T,d_out) ## (hidden,b) * (b,ouput_dim)
        
        d_ws = np.dot(self.h_t,d_out)                        ## (hidden,b) * (b,output_dim)  
        d_ht = np.dot(d_out,params_dict['W_s']['value'].T)     ## (b,output) * (hidden,output).T

        dz = np.dot(d_ht,1-np.tanh(self.z)*np.tanh(self.z))    

        d_bx =  dz.sum(axis = 0)
        d_Whh = np.dot(dz.T,self.h)  ### hidden * hidden
        d_Whx = np.dot(dz,self.x)          ### hidden * 
        d_x = bp.dot()

        

        return 0

class Rnn_Layer:
    def __init__(num_layer,input_dim,hidden_dim,output_dim): 
        """ 
        """ 
        self.num_layer = num_layer
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
    def init(self):
        self.h = np.zeros(self.hidden_layer,1)
        
        self.W_hx = np.random.rand(self.hidden_dim,self.input_dim) 
        self.W_hh = np.random.rand(self.hidden_dim,self.hidden_dim) 
        self.W_out = np.random.rand(self.output_dim,self.hidden_dim) 


    def forward(self,inputs):
        self.init() 
        for i in range(len(inputs)):
            h_t = np.tanh(W_hh*self.h + W_hx * inputs[i] + self.bx) 
            y = np.softmax(self.Ws*h + self.by)

        return y 
    
    def backward(self): 
        """ 
        Loss = y.log(y_hat) --> dL/d(y_hat) = 1/

        dL/dWs = h(y_hat - y) 
        dL/dh = Ws(y_hat - y) 
        dL/dby = 

        dh/d(Whh) = 
        dh/d(Whx) = 
        dh/d(bx)
    
        """ 
