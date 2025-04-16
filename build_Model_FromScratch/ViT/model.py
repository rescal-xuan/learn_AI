import torch
import torch.nn as nn
import torch.nn.functional as F

device ='cuda' if  torch.cuda.is_available()  else 'cpu'
    
def embed_token(img, patch_size, weight, batch_size, d_model):
    patch =F.unfold(img,patch_size,stride=patch_size).transpose(1,2) # (batch_size, num_patches, patch_size * patch_size * channel)
    img_embed = patch @ weight  #(batch_size, num_patches, d_model)
    cls_token_embed =nn.Parameter(torch.randn(batch_size,d_model).unsqueeze(1)).to(device)
    token_embed =torch.cat([img_embed,cls_token_embed],dim=1).to(device)
    pos_embed =nn.Parameter(torch.randn(batch_size,token_embed.shape[1],d_model)).to(device)
    input_token =pos_embed+token_embed.to(device)
    return input_token

class ViTransformer(nn.Module):
    def __init__(self,d_model,num_head,patch_size,weight,class_num=10 ):
        super().__init__()
        self.num_head=num_head
        self.d_model =d_model
        self.class_num=class_num
        self.weight=weight
        self.patch_size=patch_size
        self.trans = nn.TransformerEncoderLayer(d_model, nhead=8)  
        self.transformer_encoder = nn.TransformerEncoder(self.trans, num_layers=num_head)
        self.cls_liner=nn.Linear(d_model,class_num)
    def forward(self,x):
        batch_size = x.shape[0]
        
        input_token =embed_token(x,self.patch_size,self.weight,batch_size,self.d_model)
        x =self.transformer_encoder(input_token)
        cls_token =x[:,0,:]
        
        out =self.cls_liner(cls_token)
        return out
        
        
   

if __name__ == "__main__":
    batch_size, channel, height, weight = 1, 1, 224,224  # b,c,h,w
    patch_size = 2
    patch_d = patch_size * patch_size * channel
    d_model = 8
    img = torch.randn(batch_size, channel, height, weight)
    weight_matrix = torch.randn(patch_d, d_model)
    model = ViTransformer(d_model, 6, patch_size, weight_matrix)
    out = model(img)
    print(out.shape) # 预期输出 (batch_size, num_classes) 这里是 (1, 10)
   
   
   