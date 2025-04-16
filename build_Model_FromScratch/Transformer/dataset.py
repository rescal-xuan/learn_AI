import torch
from torch.utils.data  import DataLoader,Dataset
import random
from  transformers import  AutoTokenizer
def  read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
        return data
def split_data(data_path):
    data =read_data(data_path)
    train_data =data[:int(len(data)*0.95)]
    test_data =data[int(len(data)*0.95):]
    with open('./data/train.txt','w',encoding='utf-8')  as f:
        f.writelines(train_data)   
    with open('./data/test.txt','w',encoding='utf-8')  as f:
        f.writelines(test_data)  
        
def get_max_length(data_path,tokenizer):
    max_len =0
    datas =read_data(data_path)
    for data in datas:
        en,zh =data.strip().split('\t')[:2] 
    max_len = max(max_len,max(len(tokenizer(en)['input_ids']),len(tokenizer(zh)['input_ids'])))
    print(max_len)       


    
class transDataSet(Dataset):
    def __init__(self,tokenzier,data_path,max_len=512):
        super().__init__()
        self.tokenizer =tokenzier
        self.data = read_data(data_path)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        src,tgt =self.data[index].strip().split('\t')[:2]
        
        return src,tgt
    
    
   
if __name__  =="__main__":
    # split_data('./data/cmn.txt')
    tokenizer =AutoTokenizer.from_pretrained('./model_/gpt2_chinese')
    # get_max_length('./data/cmn.txt',tokenizer)
    dataset =transDataSet(tokenizer,'./data/train.txt',64)
    print(dataset[0])