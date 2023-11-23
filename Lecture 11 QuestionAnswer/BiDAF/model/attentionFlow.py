import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import argparse

class attentionFlow(nn.Module):
    """ 
    tính S : 

        S = \alpha(H,U) 
        trong đó : H(2d.T) là ma trận sinh từ context
                   U (2d.J) là ma trận sinh từ query
        paper chọn: \alpha = W^T[h;u;h.u] 
                -> W^T (6d) 
                -> . ko phải nhân mà là elementwise multi
    
    Sau đó chúng ta sử dụng S để tính attention theo cả 2 hướng : 
    context -> query 
    query -> context 

    Lại sao tại 2 hướng ??? ko phải attention 2 cái sẽ là đối xứng nhau à ?? 
    Vì với query chúng ta sẽ sử dụng tất cả các từ còn với context chúng ta \\ 
    chỉ xét một vài cụm từ quan trọng trong nó vì context thường rất dài


    """
    def __init__(self,args):
        super().__init__()
        self.batch_size = args.batch_size 
        self.hidden_size = args.hidden_size
        self.w = nn.Linear(3*args.hidden_size,1)

    def forward(self,h,u):
        """ 
        Params:
            - h : shape = (B,T,hidden_size) 
            - u : shape = (B,J,hidden_size) 
        """
        shape = (self.batch_size,h.shape[1],u.shape[1],self.hidden_size)
        h_extended = h.unsqueeze(2).expand(shape) ## Chuyển h -> (B,T,1,hidden_size)
        u_extended = u.unsqueeze(1).expand(shape) ## chuyển u -> (B,1,J,hidden_size)
        elementWiseMulti = torch.mul(h_extended,u_extended)
        concat = torch.cat((h_extended,u_extended,elementWiseMulti),3)

        S = self.w(concat).view(self.batch_size,h.shape[1],u.shape[1]) ## (B,T,J,hidden_size) -> (B,T,J,1) -> (B,T,J)
        ## Context to Query: Biểu thị những từ query phù hợp nhất với từng context
        ## Tính weight của từng từ trong context = softmax
        ## Sau đó nhân weight này với các từng từ trong query
        S_softmax  = F.softmax(S,dim = 1)
        c2q = torch.bmm(S_softmax,u) ## N,T,J * N,J,hidden_size = N,T,hidden_size
        ## Query to Context: Biểu thị từ context nào có sự tương đồng gần nhất với một trong các từ query.
        ## hiểu đơn giản là tính similar của các từ context với query.. từ context nào similiar cao nhất với 1 trong các query thì lấy 
        S_max_col, _ = torch.max(S,dim  = 2)  ## N,T   chọn ra nhưng context lớn nhất với tưng query
        b_t = F.softmax(S_max_col, dim = 1).unsqueeze(1) ## N,T -> N,1,T : tính trọng số của từng từ context đã chọn ở trên 
        q2c = torch.bmm(b_t,h).squeeze() ## N,1,T * N,T,hidden_size = N,1,hidden_Size
        q2c = q2c.unsqueeze(1).expand(self.batch_size,h.shape[1],self.hidden_size)


        ## Tính G 
        G = torch.cat([h,c2q,h.mul(c2q),h.mul(q2c)],2)

        return G ## N,T,4*hidden_size

if __name__ =='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--batch-size',default=10)
    parser.add_argument('--hidden-size',default=50)
    args = parser.parse_args() 

    layer = attentionFlow(args) 
    context = torch.rand(10,100,50) 
    query = torch.rand(10,20,50) 
    if layer(context,query).shape == (args.batch_size,context.shape[1],4*args.hidden_size):
        print("Pass")
    else:
        print("False")
