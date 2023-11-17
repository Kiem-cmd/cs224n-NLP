import argparse 
import torch 
import torch.nn as nn 

def train(args,data):
    device = torch.device("cuda" if torch.cuda.is_available() else 'gpu') 
    model = BiDaf(args, data.WORD.vocab.vectors).to(device) 
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    args = parser.parse_args()
    
    print('Loading data ---------') 
    
    data = SQuAD(args) 
    
    print('Load data completed !!')
    
    print('Training Start ..........') 
    best_model = train(args,data)
    print('Training finished !!!!!!!!!!') 
    
if __name__ == '__main__':
    main()
    