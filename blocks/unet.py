import torch
import torch.nn as nn
from blocks.conv import Conv2dModule

'''
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, num_classes, num_filters=64, activation='relu', norm='batch_norm'):
        super(UNetEncoder, self).__init__()

        # 编码器部分（Downsampling）
        self.encoder1 = UNetBlock(in_channels, num_filters, activation=activation, norm=norm)  # [64, H, W]
        self.encoder2 = UNetBlock(num_filters, num_filters * 2, activation=activation, norm=norm)  # [128, H/2, W/2]
        self.encoder3 = UNetBlock(num_filters * 2, num_filters * 4, activation=activation, norm=norm)  # [256, H/4, W/4]
        self.encoder4 = UNetBlock(num_filters * 4, num_filters * 8, activation=activation, norm=norm)  # [512, H/8, W/8]
        self.bottleneck = UNetBlock(num_filters * 8, num_filters * 16, activation=activation, norm=norm)  # [1024, H/16, W/16]

        # 分类头（输出8个类别）
        self.classifier = nn.Conv2d(num_filters * 16, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bottleneck = self.bottleneck(enc4)

        # 分类头
        output = self.classifier(bottleneck)
        return output


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', norm='batch_norm', upsample=False):
        super(UNetBlock, self).__init__()

        # 是否上采样
        self.upsample = upsample
        self.conv1 = Conv2dModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm)
        self.conv2 = Conv2dModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm)

        if self.upsample:
            # 上采样操作，使用转置卷积（反卷积）
            self.upsample_layer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip_connection=None):
        # 先进行卷积
        x = self.conv1(x)
        x = self.conv2(x)

        if self.upsample:
            # 如果是解码器部分，进行上采样
            x = self.upsample_layer(x)

        # 如果有跳跃连接（跳跃连接传递的是来自编码器部分的特征），将其拼接到当前层
        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        return x


# Example usage
if __name__ == '__main__':
    # 创建 U-Net 编码器模型，输出 8 类
    unet_encoder = UNetEncoder(in_channels=3, num_classes=8, num_filters=64)

    #设输入图像为(batch_size=8, channels=3, height=256, width=256)
    input_image = torch.randn(8, 3, 256, 256)

    # 前向传播
    output = unet_encoder(input_image)
    print(output.shape)  # 输出尺寸： (8, 8, 16, 16) 如果输出8个类别'''

import torch
import torch.nn as nn
from blocks.conv import Conv2dModule

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', norm='batch_norm', upsample=False):
        super(UNetBlock, self).__init__()
        self.upsample = upsample
        self.conv1 = Conv2dModule(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=activation,
                                  norm=norm)
        self.conv2 = Conv2dModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=activation,
                                  norm=norm)

        if self.upsample:
            self.upsample_layer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip_connection=None):
        x = self.conv1(x)
        x = self.conv2(x)

        if self.upsample:
            x = self.upsample_layer(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        return x


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, num_classes, num_filters=64, activation='relu', norm='batch_norm'):
        super(UNetEncoder, self).__init__()

        # 编码器部分（Downsampling）
        self.encoder1 = UNetBlock(in_channels, num_filters, activation=activation, norm=norm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样

        self.encoder2 = UNetBlock(num_filters, num_filters * 2, activation=activation, norm=norm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样

        self.encoder3 = UNetBlock(num_filters * 2, num_filters * 4, activation=activation, norm=norm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样

        self.encoder4 = UNetBlock(num_filters * 4, num_filters * 8, activation=activation, norm=norm)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样

        self.bottleneck = UNetBlock(num_filters * 8, num_filters * 16, activation=activation, norm=norm)

        # 分类头
        self.classifier = nn.Conv2d(num_filters * 16, num_classes, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局池化
        self.fc = nn.Linear(num_classes, num_classes)  # 全连接层

    def forward(self, x):
        # 编码器部分
        enc1 = self.encoder1(x)
        enc1_pool = self.pool1(enc1)  # 下采样

        enc2 = self.encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2)  # 下采样

        enc3 = self.encoder3(enc2_pool)
        enc3_pool = self.pool3(enc3)  # 下采样

        enc4 = self.encoder4(enc3_pool)
        enc4_pool = self.pool4(enc4)  # 下采样

        bottleneck = self.bottleneck(enc4_pool)

        # 分类头
        output = self.classifier(bottleneck)
        output = self.global_pool(output)  # [batch_size, num_classes, 1, 1]
        output = output.view(output.size(0), -1)  # [batch_size, num_classes]
        output = self.fc(output)  # [batch_size, num_classes]
        return output


# Example usage
if __name__ == '__main__':
    # 创建 U-Net 编码器模型，输出 8 类
    unet_encoder = UNetEncoder(in_channels=3, num_classes=8, num_filters=64)

    # 设输入图像为 (batch_size=8, channels=3, height=256, width=256)
    input_image = torch.randn(8, 3, 256, 256)

    # 前向传播
    output = unet_encoder(input_image)
    print(output.shape)  # 输出尺寸： (8, 8, 16, 16)
