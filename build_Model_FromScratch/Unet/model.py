import torch  
import torch.nn as nn
import torch.nn.functional  as F


import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    (卷积层 -> 批量归一化 -> ReLU) * 2
    使用更简洁的方式定义双卷积块。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)  # 使用inplace=True节省内存

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)  # 使用inplace=True节省内存

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class DownSample(nn.Module):
    """
    下采样模块：最大池化 -> 双卷积
    将最大池化层和双卷积层组合在一起进行下采样。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class UpSample(nn.Module):
    """
    上采样模块: 双线性插值 -> 拼接 -> 双卷积
    先使用双线性插值进行上采样，然后将上采样后的特征图与来自编码器的特征图拼接在一起，最后使用双卷积模块进行处理。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 调整大小以匹配 x2 (如果需要)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 拼接特征图
        x = torch.cat([x2, x1], dim=1)

        # 双卷积
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    """
    输出卷积层：1x1 卷积，将特征图转换为最终的类别预测。
    """

    def __init__(self, in_channels, num_classes):  # 更名为 in_channels，更清晰
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)  # 更名为 self.conv

    def forward(self, x):
        return self.conv(x)  # 直接返回卷积结果
        

class UNet(nn.Module):
    """
    U-Net 模型：用于图像分割任务。
    """

    def __init__(self, in_channels, num_classes, bilinear=True):
        super().__init__()  # 使用更简洁的 super() 调用

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.encoder1 = DoubleConv(in_channels, 64)  # 输入层
        self.encoder2 = DownSample(64, 128)
        self.encoder3 = DownSample(128, 256)
        self.encoder4 = DownSample(256, 512)

        factor = 2 if bilinear else 1
        self.bottleneck = DownSample(512, 1024 // factor)  # 瓶颈层

        self.decoder1 = UpSample(1024, 512 // factor)
        self.decoder2 = UpSample(512, 256 // factor)
        self.decoder3 = UpSample(256, 128 // factor)
        self.decoder4 = UpSample(128, 64)

        self.out_conv = OutConv(64, num_classes)  # 输出层

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # 瓶颈层
        bottleneck = self.bottleneck(enc4)

        # 解码器
        dec1 = self.decoder1(bottleneck, enc4)
        dec2 = self.decoder2(dec1, enc3)
        dec3 = self.decoder3(dec2, enc2)
        dec4 = self.decoder4(dec3, enc1)

        # 输出
        logits = self.out_conv(dec4)
        return logits
        
        
if __name__ == "__main__":
    # 示例用法
    in_channels = 3  # 输入图像的通道数
    num_classes = 1  # 输出的类别数（例如，二元分割问题）
    img_size = 512
    # 创建 U-Net 模型
    model = UNet(in_channels=in_channels, num_classes=num_classes)
    # 创建一个随机输入张量
    input_tensor = torch.randn(1, in_channels, img_size, img_size)
    # 进行前向传播
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # torch.Size([1, 1, img_size, img_size])