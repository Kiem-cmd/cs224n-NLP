import argparse 
import torch 
import time
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from data.dataloader import Squad
from model.bidaf import BiDAF

def train(args,dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else 'gpu') 
    model = BiDAF().to(device) 
    parameters = filter(lambda x: x.requires_grad, model.parameters())


    optimizer = optim.Adadelta(parameters,lr = args.learning_rate) 

    print("Starting trainning .....")
    start = time.time()

    model.train() 
    
    for i,batch in enumerate(dataloader):
        optimizer.zero_grad() 
        if epochs % 10 ==0:
            print("")
        context_word,question_word,context_char,quesion_char,label = batch 
        context_word,question_word,context_char,question_char = context_word.to(device),question_word.to(device),context_char.to(device),quesion_char.to(device)
        predict = model(context_word,context_char,question_word,quesion_char)
        start_predict, end_predict = predict
        start = label[:,0]
        end = label[:,1]
        loss = F.cross_entropy(start_predict,start) + F.cross_entropy(end_predict,end)
        loss.backward()
        optimizer.step()

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
    