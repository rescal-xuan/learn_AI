import torch
from torch import nn
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len,d_model):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        pos =torch.arange(0,max_seq_len).unsqueeze(1)
        _2i =torch.arange(0,d_model,2)
        div_item =10000**(_2i/d_model)
       
        pe =torch.zeros(max_seq_len,d_model)
        pe[:,0::2] =torch.sin(pos/div_item)
        pe[:,1::2] =torch.cos(pos/div_item)

        # plt.matshow(pe)
        # plt.show()
        
        pe =pe.unsqueeze(0)
        self.register_buffer("pe",pe)
    def forward(self, x):
        
        x = x + self.pe[:, :x.size(1),:]
        return x
        

def attention(query,key,value,mask=None):
    d_k =key.size(-1)
    att_ =  torch.matmul(query, key.permute(0, 1, 3, 2)) /d_k**0.5
    if mask is not None:
        att_ =att_.masked_fill(mask==0,1e-9)
    att_score =torch.softmax(att_,-1)
    return torch.matmul(att_score,value)

class MutiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert  d_model%heads==0 
        # 保存模型维度和头数
        self.d_model = d_model
        self.d_k = d_model // heads  # 每个头对应的维度
        self.h = heads  # 头的数量

        # 初始化线性层，用于将输入转换为查询（Q）、键（K）和值（V）
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        # 初始化Dropout层，用于正则化
        self.dropout = nn.Dropout(dropout)
        # 初始化输出线性层，用于将多头注意力输出转换为模型维度
        self.out = nn.Linear(d_model, d_model)

    # 定义注意力机制的计算过程
    def attention(self, q, k, v, mask=None):
        # 计算Q和K的矩阵乘积，然后除以根号下d_k
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k**0.5
        
        # 如果提供了掩码，则将掩码对应的位置设置为负无穷，这样在softmax后这些位置的值为0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 应用softmax函数获得注意力权重
        scores = torch.softmax(scores, dim=-1)
        # 应用dropout
        scores = self.dropout(scores)
        # 将注意力权重和V相乘得到输出
        output = torch.matmul(scores, v)
        return output

    # 定义前向传播过程
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 将输入Q、K、V通过线性层，并调整形状以进行多头注意力计算
        q = self.q_linear(q).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # 计算注意力输出
        scores = attention(q, k, v, mask)
        # 将多头输出合并，并调整形状以匹配模型维度
        concat = scores.transpose(1, 2).contiguous().reshape(batch_size, -1, self.d_model)
        # 通过输出线性层
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.2):
        super().__init__()  
        # self.fc1 = nn.Linear(d_model, d_ff)  
        # self.fc2 = nn.Linear(d_ff, d_model)   
        # self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        
        self.ffn =nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model),
            nn.ReLU(),
            nn.Dropout(dropout) ,
        )
    def forward(self,x):
        # x= self.fc1(x)
        # x =self.relu(x)
        # x=self.fc2(x)
        # x =self.dropout(x)
        # return x
        return self.ffn(x)

class encodelayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout=0.2):
        super().__init__()
        self.mutiheadattention = MutiHeadAttention(heads, d_model,dropout)
        self.fnn = FeedForward(d_model, d_ff, dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        muti_att_out = self.mutiheadattention(x, x, x, mask)
        # 改进：先残差连接，再层归一化
        # print(muti_att_out.shape,x.shape)
        muti_att_out = self.norm[0](muti_att_out + x)  # 使用 norm1

        fnn_out = self.fnn(muti_att_out)
        # 改进：先残差连接，再层归一化
        fnn_out = self.norm[1](muti_att_out + fnn_out)  # 使用 norm2
        out = self.dropout(fnn_out)
        return out

class encoder(nn.Module):
    def __init__(self, vocab_size, heads, d_model, d_ff, num_layer,  max_seq_len=512):
        super().__init__()
        # print(vocab_size.shape,d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(max_seq_len, d_model)
        self.encode_layer = nn.ModuleList(
            [encodelayer(heads, d_model, d_ff) for _ in range(num_layer)]
        )

    def forward(self, x, src_mask):
        embed_x = self.embedding(x)
        pos_x = self.position(embed_x)
        for layer in self.encode_layer:
            pos_x = layer(pos_x, src_mask)
        return pos_x
    
   
class decodelayer(nn.Module):
    def __init__(self,heads,d_model,d_ff,dropout=0.2):
        super().__init__()   
        self.mask_muti_att =MutiHeadAttention(heads,d_model,dropout)
        self.muti_att =MutiHeadAttention(heads,d_model,dropout)
        self.norms =nn.LayerNorm(d_model)
        self.ffn =FeedForward(d_model,d_ff)
        self.dropout =nn.Dropout(dropout)
    def forward(self,x,encode_kv,mask=None,src_mask=None):
        mask_muti_att_out  =self.mask_muti_att(x,x,x,mask)
        mask_muti_att_out =self.norms(x+mask_muti_att_out)
        muti_att_out =self.muti_att(mask_muti_att_out,encode_kv,encode_kv,src_mask)
        muti_att_out =self.norms(mask_muti_att_out+muti_att_out)
        ffn_out =self.ffn(muti_att_out)
        ffn_out = self.norms(ffn_out+muti_att_out)
        out =self.dropout(ffn_out)
        
        return out 
 
class decoder(nn.Module):
    def __init__(self,vocab_size,heads,d_model,d_ff,num_layer, padding_idx,max_seq_len=512):
        super().__init__()
        
        self.embedding =nn.Embedding(vocab_size,d_model, padding_idx)
        self.position =PositionalEncoding(max_seq_len,d_model)
        self.decode_layer =nn.ModuleList(
            [decodelayer(heads,d_model,d_ff) for _  in range( num_layer)]
        )
    def forward(self,x,encode_kv,src_mask, tgt_mask):
        embed_x =self.embedding(x)
        pos_x =self.position(embed_x)
        for layer  in self.decode_layer:
            pos_x=layer(pos_x,encode_kv,src_mask, tgt_mask)
        return pos_x

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,heads,d_model,d_ff,num_layer,max_seq_len=512, dropout=0.2):
        super().__init__()
        self.encoder =encoder(src_vocab_size,heads,d_model,d_ff,num_layer,max_seq_len)
        self.decoder =decoder(tgt_vocab_size,heads,d_model,d_ff,num_layer,max_seq_len)
        self.liner =nn.Linear(d_model,tgt_vocab_size)
        self.dropout=nn.Dropout(dropout)
        # self.padding_idx =padding_idx
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
        
    def forward(self, src, tgt):
        
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        output = self.liner(dec_output)
        return output
         

   
if __name__ =="__main__":  
        
    src = torch.randint(0,1000,(10,512))
    tgt= torch.randint(0,1000,(10,512))  
    model =Transformer(1000,1000,8,512,2048,6)
    out =model(src,tgt)
    print(out)