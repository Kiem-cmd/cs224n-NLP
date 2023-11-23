## link: 
## https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json   (40MB)
## https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json     (4MB)  
import numpy as np 
import os 
import urllib.request 
from tqdm import tqdm 
import argparse




def download_url(url, output_path, show_progress = True):
    if show_progress:
        with tqdm(unit = 'B', unit_scale= True, miniters = 1, desc = url.split('/')[-1]) as t: 
            urllib.request.urlretrieve(url, 
                                        filename = output_path,
                                        reporthook= t.update()) 

    else:
        urllib.request.urlretrieve(url, output_path) 
if __name__ == '__main__':
    url_default_train ="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    url_default_dev = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",default= default_train,type = str) 
    parser.add_argument("--output_path",default="../BiDAF/data/squad/",type = str) 
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    output_name = args.url.split("/")[-1]
    print("Downloading ........ ")
    download_url(args.url,args.output_path+output_name)
    print(f"Download complete ...save to {args.output_path+output_name}")