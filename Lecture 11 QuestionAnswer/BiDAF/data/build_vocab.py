import json 
import os 
import pandas as pd 
import torch 
import time 
from tqdm import tqdm
from torchtext import data 
from torchtext import datasets
from torchtext.vocab import vocab 
from collections import Counter,OrderedDict
from torch.utils.data import Dataset 
import spacy 
import argparse
from torchtext.data.utils import get_tokenizer 
import pickle
tokenizer = get_tokenizer("basic_english") 


"""
Data file json: 
article: 
|--- title 
|--- paragraphs
       |--- context 
       |--- qas 
            | -- question 
            | -- id 
            | -- answers 
                    |--- anwser_start 
                    |--- text

..... 

|--- paragraphs: 
        |--- 
        |---



"""
def parser_data(data):
    '''
    
    
    '''
    print("Parsering from json to DataFrame ............")
    start = time.time()
    data = data['data'] 
    qa_list = [] 
    for paragraphs in data:
        for para in paragraphs['paragraphs']:
            context = para['context'] 
            for qa in para['qas']:
                id = qa['id'] 
                question = qa['question']
                for ans in qa['answers']:
                    answer = ans['text']
                    ans_start = ans['answer_start']
                    ans_end = ans_start + len(answer)

                    qa_dict = dict() 
                    qa_dict['id'] = id 
                    qa_dict['context'] = context
                    qa_dict['question'] = question
                    qa_dict['answer'] = answer 
                    qa_dict['ans_start'] = ans_start 
                    qa_dict['ans_end'] = ans_end 
                    qa_list.append(qa_dict) 
    end = time.time() 
    print("Number of Q/A: ",len(qa_list))
    print(f"Parser data from json to DataFrame in {end- start}s")
    print("--------------------------------------------------------------------")
    return pd.DataFrame(qa_list) 


def text2vocab(data):
    ''' 
    
    
    
    '''
    print("Building text vocab ...............")
    text = [] 
    total = 0 
    start = time.time()
    for paragraphs in data:
        context_unique = list(paragraphs.context.unique())
        question_unique = list(paragraphs.question.unique()) 
        text.extend(context_unique)
        text.extend(question_unique) 
    print("Sum of context + question: ",len(text)) 
    end = time.time()
    print(f"Build text vocab in {end-start}s")
    print("-------------------------------------------------------------------")
    return text 

def build_word_vocab(vocab_text):
    '''
    
    
    '''
    print("Building word vocab ..................")
    start = time.time()
    words = []
    for seq in vocab_text:
        words.extend(tokenizer(seq))
    word_counter = Counter(words) 
    sorted_by_freq = sorted(word_counter.items(), key = lambda x:x[1],reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq)
    vocab_ = vocab(ordered_dict,specials=['<unk>'])
    end = time.time() 
    print(f"Len word vocab: {vocab_.__len__()}")
    print(f"Build word vocab in: {end - start}s")
    print("-------------------------------------------------------------------")
    return vocab_

def build_char_vocab(vocab_text):
    ''' 
    
    
    '''
    print("Building char vocab ..........") 
    start = time.time() 
    chars = []
    for seq in vocab_text:
        for ch in seq:
            chars.append(ch) 
    char_counter = Counter(chars)
    sorted_by_freq = sorted(char_counter.items(), key = lambda x:x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq)
    vocab_ = vocab(ordered_dict,min_freq=20,specials = ['<unk>','<pad>'])
    print(f"Len char vocab: {vocab_.__len__()}")
    end = time.time()
    print(f"Build char vocab in: {end - start}s")
    print("-------------------------------------------------------------------")
    return vocab_
def convert_idx(text,ans_start,ans_end):
    current = 0
    spans = []
    answer_span = []
    text = text.replace('"',"'").lower()
    tokens = tokenizer(text)
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    for idx,span in enumerate(spans):
        if not (ans_end <= span[0] or ans_start >= span[1]):
            answer_span.append(idx)
    y1,y2 = answer_span[0],answer_span[-1]
    return (y1,y2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--path",default = '/home/kiem/Github/cs224n-NLP/Lecture 11 QuestionAnswer/BiDAF/data/squad/train-v1.1.json')
    args = parser.parse_args()
    with open(args.path, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    df = parser_data(data)
    text_vocab = text2vocab([df])
    word_vocab = build_word_vocab(text_vocab)
    char_vocab = build_char_vocab(text_vocab)
    print("Processing data to dataFrame......")
    word_pipeline = lambda x: word_vocab(tokenizer(x))
    char_pipeline = lambda x: char_vocab([i for i in x]) 
    df['context_word'] = df['context'].apply(word_pipeline)
    df['question_word'] = df['question'].apply(word_pipeline)
    df['label'] = [convert_idx(x['context'],x['ans_start'],x['ans_end']) for _,x in df.iterrows()]
    df.to_pickle('../BiDAF/data/squad/train.pkl')
    print("Saved dataframe to train.pkl")
    print("Saving.............")
    with open('../BiDAF/data/squad/wordVocab.pickle', 'wb') as f:
        pickle.dump(word_vocab, f)
    with open('../BiDAF/data/squad/charVocab.pickle', 'wb') as f:
        pickle.dump(char_vocab, f)
    print("Saved to wordVocab.pkl & charVocab.pkl")