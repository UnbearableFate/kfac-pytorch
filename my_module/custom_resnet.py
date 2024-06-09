import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader


class ResNetForCIFAR10(nn.Module):
    def __init__(self, layers = 18):
        super(ResNetForCIFAR10, self).__init__()
        # 加载预训练的resnet模型
        if layers == 18:
            self.model = models.resnet18()
        if layers == 34:
            self.model = models.resnet34()
        if layers == 50:
            self.model = models.resnet50()
        if layers == 101:
            self.model = models.resnet101()
        # 修改第一层卷积核大小和步长
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 修改最后的全连接层，以适应 CIFAR-10 的10个类别
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        self.model_name = f"resnet{layers}"

    def forward(self, x):
        return self.model(x)
    

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

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet18, self).__init__()
        # 加载预训练的resnet18模型
        self.resnet18 = models.resnet18()
        # 替换最后的全连接层以匹配FashionMNIST的10个类别
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet18(x)

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整为224x224
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道伪彩色图像
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import torch.nn as nn
import torchvision.transforms as transforms


class MLP(nn.Module):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    def __init__(self, input_size=28 * 28, hidden_size=64, num_hidden_layers=5, output_size=10):
        super(MLP, self).__init__()
        layers = [nn.Flatten()]

        # Add the first layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Add hidden layers
        for _ in range(num_hidden_layers - 1):  # subtract 1 because the first layer is already added
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Add the output layer
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)