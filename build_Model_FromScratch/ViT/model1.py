import torch
import torch.nn as nn
import torch.nn.functional as F
from module.ECA   import  ECA_layer  as  eca

class PatchEmbed(nn.Module):
    """
    将图像分割成 Patches 并进行嵌入的模块。

    Args:
        img (torch.Tensor): 输入图像，形状为 (批大小, 通道数, 高度, 宽度)。
        patch_size (int): 每个 Patch 的大小 (正方形)。
        d_model (int): 嵌入维度，即每个 Patch 嵌入后的向量长度。
    """
    def __init__(self, patch_size, d_model, in_channels):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.in_channels = in_channels
        self.patch = nn.Conv2d(in_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        batch_size, in_channels, height, width = img.shape  # 获取输入图像的形状

        # 使用卷积层进行 Patch 嵌入
        conv_embed = self.patch(img)  # (批大小, d_model, height/patch_size, width/patch_size)

        # 将卷积输出展平并调整维度
        patch_embed = conv_embed.flatten(2).transpose(1, 2)  # (批大小, (height/patch_size) * (width/patch_size), d_model)

        return patch_embed

class ViTransformer(nn.Module):
    def __init__(self, d_model, img_size, patch_size, cls_num, in_channels=3, num_head=8, num_layers=6): #添加了in_channels, num_layers参数
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2  # Patch 的数量
        self.d_model = d_model
        self.in_channels = in_channels  # 输入通道数
        self.patch_embed = PatchEmbed(patch_size, d_model, in_channels)  # 使用 PatchEmbed 类
        self.norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # (1, 1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.d_model))  # (1, num_patches + 1, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, num_head), num_layers) # 使用num_layers
        self.cls_liner = nn.Linear(d_model, cls_num)
        self.eca =eca(channel=3)
    def forward(self, img):
        batch_size = img.size(0)  # 获取批大小

        # Patch Embedding
        patch_embeddings = self.patch_embed(img)  # (批大小, num_patches, d_model)

        # Class Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (批大小, 1, d_model)
        # Concatenate Class Token with Patch Embeddings
        x = torch.cat((cls_tokens, patch_embeddings), dim=1)  # (批大小, num_patches + 1, d_model)

        # Positional Embedding
        x = x + self.pos_embedding  # (批大小, num_patches + 1, d_model)
        x = self.norm(x)

        # Transformer Encoder
        x=self.eca(x)
        out = self.transformer(x.transpose(0, 1)).transpose(0, 1)  # (批大小, num_patches + 1, d_model)

        # Classification
        out = self.cls_liner(out[:, 0, :])  # (批大小, cls_num)
        return out
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    batch_size = 1
    in_channels = 3
    height = 224
    width = 224
    patch_size = 16
    d_model = 768
    img = torch.randn(batch_size, in_channels, height,width)
    # patch_embed(img,patch_size,d_model)
    model = ViTransformer(d_model, img.shape[3],patch_size,10,3)
    out = model(img)
    print(out.shape) # 预期输出 (batch_size, num_classes) 这里是 (1, 10)