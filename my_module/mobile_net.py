import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

class CustomMobileNetV3Small(nn.Module):

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

    def __init__(self, num_classes=10):
        super(CustomMobileNetV3Small, self).__init__()
        # 加载预训练的 MobileNetV3 Small 模型
        self.model = models.mobilenet_v3_small(pretrained=False)
        # 修改最后的分类层
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

    def forward(self, x):
        return self.model(x)
