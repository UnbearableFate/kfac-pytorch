import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
from torchvision.models import shufflenet_v2_x0_5
from torchvision.models import alexnet
from my_module.custom_resnet import ResNetForCIFAR10 ,SimpleCNN
import os
from torch.utils.data import DataLoader

import kfac
from my_module.mobile_net import CustomMobileNetV3Small
import logging

from my_module.model_split import ModelSplitter

logging.basicConfig(level=logging.NOTSET)

today = "0923"
DATA_DIR = ""
LOG_DIR = ""
Share_DIR = ""
if os.path.exists("/home/yu"):
    DATA_DIR = "/home/yu/data"
    LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs"+today
    Share_DIR = "/home/yu/workspace/kfac-pytorch/share_files"
elif os.path.exists("/Users/unbearablefate"):
    DATA_DIR = "/Users/unbearablefate/workspace/data"
    LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs"+today
    Share_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/share_files"
elif os.path.exists("/work/NBB/yu_mingzhe/kfac-pytorch"):
    DATA_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data"
    LOG_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/runs"+today
    Share_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data/share_files"

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def replace_relu_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_relu_inplace(child)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整 CIFAR-10 图片大小
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform2 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=DATA_DIR+"/CIFAR10", train=True, download=False, transform=transform2)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=DATA_DIR+"/CIFAR10", train=False, download=False, transform=transform2)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

# 加载 MobileNetV3 模型并进行修改以适应 CIFAR-10
#model = alexnet(num_classes = 10)
#replace_relu_inplace(model)

#model = ModelSplitter(model, 128)
model = SimpleCNN()
model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

preconditioner = kfac.preconditioner.KFACPreconditioner(model=model)

# 训练模型
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    for epoch in range(10):  # 训练10个epoch
        index = 1 
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            preconditioner.step()
            optimizer.step()

            if index % 20 == 0:
                print(f'Epoch {epoch+1} {index}, Loss: {loss.item()}')
            index += 1

# 测试模型
def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')

# 训练和测试

if __name__ == '__main__':
    train(model, trainloader, criterion, optimizer, device)
    test(model, testloader, device)
