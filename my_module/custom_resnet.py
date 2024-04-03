import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

class CustomResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet34, self).__init__()
        # 加载预训练的resnet18模型
        self.resnet34 = models.resnet34()
        # 替换最后的全连接层以匹配FashionMNIST的10个类别
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet34(x)

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整为224x224
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道伪彩色图像
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
