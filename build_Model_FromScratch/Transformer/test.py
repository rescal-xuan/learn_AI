import torch
from  torch import nn,optim
from model import Transformer
from  dataset import *
from tqdm import tqdm
from transformers   import  AutoTokenizer
import os 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

tokenzier = AutoTokenizer.from_pretrained('./model_/gpt2_chinese')
src_vocab_size = len(tokenzier)
tgt_vocab_size = len(tokenzier)
heads = 8
d_model = 512
d_ff = 1024
num_layer = 6
pad_idx = tokenzier.pad_token_id
batch_size =1
max_length=64
max_seq_len =64
# 6. 初始化模型
model = Transformer(src_vocab_size, tgt_vocab_size,heads,d_model, d_ff, num_layer, max_seq_len=max_seq_len).to(device) 

if os.path.exists('./model.pt'):
    model.load_state_dict(torch.load('./model.pt'))
    

input_ = 'Stop!'

input_in = tokenzier(input_, max_length=64, padding="max_length", truncation=True, return_tensors='pt')['input_ids']

input_in =input_in.to(device)

# print(input_in)
de_in = torch.ones((batch_size, max_seq_len), dtype=torch.long).to(device)

model.eval()
result =[] 
with torch.no_grad():
    for i in range(1,de_in.size(1)):
        out =model(input_in,de_in)
        pred =torch.argmax(out,-1)
         
for i  in pred:
    result.append(tokenzier.decode(i))
        
print(result)
        
    




