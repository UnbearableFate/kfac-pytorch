import torchvision
import torchvision.transforms as transforms

# 数据增强和预处理操作，这里以最基本的 ToTensor() 为例
transform = transforms.Compose([
    transforms.ToTensor()
])

# 下载和加载训练集
trainset = torchvision.datasets.CIFAR10(
    root='/work/xg24i002/x10041/data',            # 数据存储路径
    train=True,               # 是否为训练集
    download=True,            # 如果本地没有数据，则自动下载
    transform=transform       # 预处理/变换
)

# 下载和加载测试集
testset = torchvision.datasets.CIFAR10(
    root='/work/xg24i002/x10041/data',
    train=False,
    download=True,
    transform=transform
)