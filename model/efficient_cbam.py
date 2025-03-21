import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

# 整合 EfficientNet-B0 和 CBAM 的模型结构
class EfficientCBAMResNet(nn.Module):
    def __init__(self, num_classes=208):
        super(EfficientCBAMResNet, self).__init__()

        # 提取 EfficientNet-B0 的特征部分（输出通道为 1280）
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.features = self.efficientnet.extract_features

        # 使用 Bottleneck + CBAM 构建后部模块
        self.resnet_cbam_layers = nn.Sequential(
            BottleneckCBAM(1280, 512, stride=2),  # 修复通道数错误：EfficientNet 输出为 1280
            BottleneckCBAM(512, 512),
            BottleneckCBAM(512, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)  # EfficientNet 特征提取
        x = self.resnet_cbam_layers(x)  # CBAM 后部增强模块
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # 最终分类输出
        return x

# CBAM：通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

# CBAM：空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接通道维度
        x = self.conv1(x)
        return self.sigmoid(x)

# Bottleneck 残差结构 + CBAM 模块
class BottleneckCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else None

    def forward(self, x):
        residual = x  # 残差连接

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.ca(out) * out  # 加入通道注意力
        out = self.sa(out) * out  # 加入空间注意力

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out