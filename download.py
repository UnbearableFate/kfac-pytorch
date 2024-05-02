import torchvision
import os

# Set the download path
download_path = '/home/yu/data/CIFAR10'

# Create the download directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Download the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root=download_path, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root=download_path, train=False, download=True)

print("CIFAR-10 dataset downloaded successfully!")