from torchvision.datasets import CIFAR10
from PIL import Image
import os

def convert_cifar10_to_imagefolder(root_out="~/data/cifar10_imgfolder"):
    os.makedirs(root_out, exist_ok=True)
    for split in ["train", "test"]:
        is_train = split == "train"
        dataset = CIFAR10(root="~/data/CIFAR10", train=is_train, download=False)
        for idx, (img, label) in enumerate(dataset):
            class_name = dataset.classes[label]
            class_dir = os.path.join(root_out, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f"{idx}.png"))

convert_cifar10_to_imagefolder()