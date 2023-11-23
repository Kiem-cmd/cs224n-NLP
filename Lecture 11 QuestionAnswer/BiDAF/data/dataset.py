import torch 
import torch.utils.data as data 
from tqdm import tqdm 
import numpy as np 




class SQuAD(data.Dataset): 
    def __init__(self,data_path):
        dataset = np.load(data_path) 
        self.context_words = 
        self.context_chars = 
        self.question_words = 
        self.question_chars = 
        self.answer_start_ind = 
        self.answer_end_ind = 


def collate_fn(batch):
    """ 
    Tạo batch: 
    context_word - shape = (batch,seq_len, ...)  -> mỗi context có số word khác nhau nên cần merge1d 
    context_char - shape = (batch,seq_len,word_len,....) -> mỗi context có số word khác nhau, mỗi word có số char khác nhau -> cần merge2d 
    
    Tương tự với question  


    """ 
    def merge_0d(scalar, dtype = torch.int64):
        return torch.tensor(scaler, dtype = dtype) 
    def merge_1d(array, dtype = torch.int64, pad = 0):
        length = [(a != pad).sum() for a in array] 
        padded = torch.zeros(len(array), max(lengths), dtype= dtype) 

        for i, seq in enumerate(array):
            end = lengths[i] 
            padded[i,:end] = seq[:end] 
        return padded 
    def merge2d(matrix, dtype = torch.int64, pad = 0):
        height = [(m.sum(1) != pad).sum() for m in matrix] 
        width = [(m.sum(0) != pad).sum() for m in matrix] 
        padded = torch.zeros(len(matrix), max(height), max(width), dtype = dtype) 

        for i,seq in enumerate(matrix):
            height, width = height[i], width[i] 
            padded[i,:height,:width] = seq[:height,:width] 

        return padded 

    context_word, context_char, \
    question_word, question_char, \
    answer_start_ind, answer_end_ind, idx = zip(*batch) 

    context_word = merge_1d(context_word) 
    question_word = merge_1d(question_word) 
    context_char = merge2d(context_char) 
    question_char = merge2d(question_char) 
    answer_start_ind = merge_0d(answer_start_ind) 
    answer_end_ind = merge_0d(answer_end_ind) 
    idx = merge_0d(idx) 

    return(context_word,context_char,question_word,question_char,answer_start_ind,answer_end_ind,idx) 

 
def main():
    print("Building dataset ................") 
    train_dataset = SQuAD() 
    train_dataloader = DataLoader()
    dev_dataloader = DataLoader() 

if __name__ =='__main__':
    main()