import json 
import os 
import nltk 
import torch 
from tqdm import tqdm
from torchtext import data 
from torchtext import datasets 
from torchtext.vocab import GloVe 
from torch.utils.data import Dataset 

import matplotlib.pyplot as plt 

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