import torch 
import pickle
import pandas as pd 
from torchtext.data.utils import get_tokenizer 





class Squad():
    def __init__(self,data,batch_size,char_vocab_path):
        self.batch_size = batch_size
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.data = data 
        self.tokenizer = get_tokenizer('basic_english')
        with open(char_vocab_path,'rb') as f:
            self.char_vocab = pickle.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index]
    def make_char_vector(self,sentence,max_seq,max_word_ctx): 
        tokens = self.tokenizer(sentence)
        char_vec = torch.ones(max_seq,max_word_ctx).type(torch.LongTensor)
        for i, word in enumerate(tokens):
            for j, ch in enumerate(word):
                char_vec[i][j] = self.char_vocab[ch]
        return char_vec
    def __iter__(self):
        for batch in self.data:
            max_seq = max([len(ctx) for ctx in batch['context_word']]) 
            padded_context_word =torch.LongTensor(len(batch),max_seq).fill_(1)
            for i,ctx in enumerate(batch['context_word']):
                padded_context_word[i,:len(ctx)] = torch.LongTensor(ctx)
            max_word_ctx = [[len(i) for i in self.tokenizer(context)]for context in batch['context']]
            max_word_ctx = max([max(i) for i in max_word_ctx])
            padded_context_char = torch.ones(len(batch),max_seq,max_word_ctx).type(torch.LongTensor)


            for i, context in enumerate(batch['context']):
                padded_context_char[i] = self.make_char_vector(context,max_seq,max_word_ctx) 
            
            max_seq_question = max([len(question) for question in batch['question']]) 
            padded_question_word = torch.LongTensor(len(batch),max_seq_question).fill_(1)
            for i,q in enumerate(batch['question_word']):
                padded_question_word[i,:len(q)] = torch.LongTensor(q) 
            max_word_question = [[len(i) for i in self.tokenizer(question)] for question in batch['question']]
            max_word_question = max([max(i) for i in max_word_question])
            padded_question_char = torch.ones(len(batch),max_seq_question,max_word_question).type(torch.LongTensor)
            for i,question in enumerate(batch['question']):
                padded_question_char[i] = self.make_char_vector(question,max_seq_question,max_word_question) 
            
            label = torch.LongTensor(list(batch['label']))

            yield padded_context_word,padded_context_char,padded_question_word,padded_question_char,label

if __name__ == '__main__':
    data = pd.read_pickle('../BiDAF/data/squad/train.pkl')
    print("Building dataloader .................")
    path = '../BiDAF/data/squad/charVocab.pickle'
    dataloader = Squad(data,16,path)
    print("Example: ")
    for i in dataloader:
        a,b,c,d,e = i
        print(a.shape)
    print("Sucessful !!!!!!!!!!!!!!!!!!")