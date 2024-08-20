from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.mobilenetv3 import InvertedResidualConfig, MobileNetV3

bneck_conf = partial(InvertedResidualConfig, width_mult=0.5)
adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=0.5)
reduce_divider = 2
dilation = 1

inverted_residual_setting = [
    bneck_conf(16, 3, 16, 16, True, "RE", 1, 1),  # C1 origin = 2 ,1 change by gpt
    bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
    bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
    bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
    bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
    bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
    bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
    bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
    bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 1, dilation),  # C4 origin = 2 ,1 change by gpt
    bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
    bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
]

last_channel = adjust_channels(1024 // reduce_divider)  # C5

cifar10_transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.Resize(224),  # 调整大小到224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
    ])

cifar10_transform_test = transforms.Compose([
        transforms.Resize(224),  # 调整大小到224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
    ])

class CustomMobileNetV3Small(nn.Module):
    
    def __init__(self, num_classes=10):
        super(CustomMobileNetV3Small, self).__init__()
        # 加载预训练的 MobileNetV3 Small 模型
        self.model = models.mobilenet_v3_small(weights = None , num_classes=10)
        # 修改最后的分类层
        self.model.classifier[1].inplace = False
        self.model.classifier[2].inplace = False
        
    def forward(self, x):
        return self.model(x)

class CustomMiniMobileNetV3ForCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomMiniMobileNetV3ForCIFAR10, self).__init__()
        # super mini MobileNetV3 Small 模型
        self.model = MobileNetV3(inverted_residual_setting, last_channel, num_classes=num_classes)
        # 修改最后的分类层
        self.model.classifier[1].inplace = False
        self.model.classifier[2].inplace = False
        
    def forward(self, x):
        return self.model(x)
