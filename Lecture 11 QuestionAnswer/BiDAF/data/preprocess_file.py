import json 
import os 
import nltk 
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
nlp = spacy.load('en') 



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
    for sequence in vocab_text: 
        for word in nlp(sequence,disable=['parser','tagger','ner']):
            words.append(word.text)
    word_counter = Counter(words) 
    sorted_by_freq = sorted(word_counter.item(), key = lambda x:x[1],reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq)
    vocab_ = vocab(ordered_dict,specials=['<unk>'])
    vocab_.set_default_index(vocab_['<unk>'])
    end = time.time() 
    print(f"Build word vocab in: {end - start}s")
    print("-------------------------------------------------------------------")
    return vocab_

def build_char_vocab(vocab_text):
    '''
    
    
    
    '''
    print('Building char vocab ................')
    start = time.time()
    chars = [] 
    for sequence in vocab_text: 
        for char in sequence: 
            chars.append(char) 
    char_counter = Counter(chars) 
    sorted_by_freq = sorted(char_counter,key = lambda x: x[1], reverse= True) 
    ordered_dict = OrderedDict(sorted_by_freq) 
    vocab_ = vocab(ordered_dict,specials=['<unk>'])
    vocab_.set_default_index(vocab_['<unk>'])
    end = time.time() 
    print('Build char vocab in ',end - start)
    return vocab_ 






def word_token(text):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(text)]


def text2token_idx(text,tokens):
    current = 0 
    out = [] 

    for token in tokens:
        current = text.find(token,current)
        if current < 0:
            raise Exception(f"Token {token} not found in text !!!!!!!! ")
        out.append((current,current+len(token)))
        current += len(token) 
    return out
def build_vocab(tokens_list):
    return vocab
def preprocess_file(file_path): 
    print(f"Pre-processing file data ............") 
    df = []
    df_ = [] 
    stt = 0 
    with open(file_path,"r",encoding="utf8") as f: 
        data = json.load(f) 
        data = data['data'] 
        for article in tqdm(data):
            for para in article['paragraphs']:
                context = para['context'].replace("''",'"').replace('``','"')
                context_tokens = word_token(context) 
                context_chars = [list(token) for token in context_tokens] 
                spans = text2token_idx(context,context_tokens)

                for qa in para['qas']:
                    stt +=1 
                    question = qa['question'].replace("''",'"').replace("``",'"')
                    question_tokens = word_token(question) 
                    question_chars = [list(token) for token in question_tokens] 

                    answer_text = qa['answers']['text'] 
                    answer_start_ind = qa['answers']['answer_start'] 
                    answer_end_ind = answer_start_ind + len(answer_text)
                    answer_list.append(answer_text) 

                    answer_span = [] 
                    for idx,span in enumerate(spans):
                        if not(answer_end_ind <= span[0] or answer_start_ind >= span[1]):
                            answer_span.append(idx) 
                    answer_token_start = answer_span[0] 
                    answer_token_end = answer_span[-1] 

                    example  = { 
                            "context_tokens": context_tokens,
                            "context_chars": context_chars,
                            "question_tokens": question_tokens,
                            "question_chars": question_chars, 
                            "answer_token_start_list": answer_token_start,
                            "answer_token_end_list": answer_token_end,
                            "stt_question": stt
                    }
                    root_examples[str(stt)] = {"context": context,
                                                 "question": question,
                                                 "spans": spans,
                                                 "answers": answer_text,
                                                 "uuid": qa["id"]}
                    df.append(example)
                    df_.append(root_examples) 
    return df,df_

if __name__ == '__main__':
    # data = preprocess_file("C:/Users/ACER/Desktop/New folder/CS224n-NLP/Lecture 11 QuestionAnswer/BiDAF/data/squad/train-v1.1.json")
    # with open('data.json', 'w') as file:
    #     json.dump(data, file)

    draw()