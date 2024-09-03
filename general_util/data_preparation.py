import os.path
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, BatchSampler

from torch.utils.data import Sampler
import numpy as np
from enum import Enum



class SimpleNonIIDSampler(Sampler):
    def __init__(self, dataset, world_size, rank):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        # 数据集按标签排序
        self.sorted_indices = self.sort_dataset_by_labels()

        # 根据 world_size 划分数据集
        self.partitioned_indices = self.partition_dataset()

    def sort_dataset_by_labels(self):
        # 获取标签并排序
        labels = np.array(self.dataset.targets)
        sorted_indices = labels.argsort()
        return sorted_indices

    def partition_dataset(self):
        # 分割数据集
        num_samples = len(self.sorted_indices) // self.world_size
        partitions = [self.sorted_indices[i * num_samples: (i + 1) * num_samples] for i in range(self.world_size)]
        return partitions

    def __iter__(self):
        # 打乱当前节点的数据
        np.random.seed(self.epoch)  # 保证每个 epoch 打乱方式一致
        indices = self.partitioned_indices[self.rank]
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.partitioned_indices[self.rank])

    def set_epoch(self, epoch):
        # 设置 epoch，以便重新打乱
        self.epoch = epoch


cifar10_transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ],
)

cifar10_transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ],
)

class DataPreparer:
    class DatasetName(Enum):
        MNIST = "MNIST"
        FashionMNIST = "FashionMNIST"
        CIFAR10 = "CIFAR10"

    train_transform_dict = {
        "MNIST": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
        "FashionMNIST": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
        "CIFAR10": cifar10_transform_train,
    }

    test_transform_dict = {
        "MNIST": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
        "FashionMNIST": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
        "CIFAR10": cifar10_transform_test,
    }

    dataset_func = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "CIFAR10": datasets.CIFAR10,
    }

    def __init__(self, data_path_root, dataset_name, world_size, rank, batch_size=64, sampler=None, train_transform =None,test_transform=None):
        self.data_path = os.path.join(data_path_root, dataset_name)
        if train_transform is not None:
            self.train_transform = train_transform
        else:
            self.train_transform = DataPreparer.train_transform_dict[dataset_name]
        if test_transform is not None:
            self.test_transform = test_transform
        else:
            self.test_transform = DataPreparer.test_transform_dict[dataset_name]

        print(f"data path : {self.data_path}")
        self.train_dataset = DataPreparer.dataset_func[dataset_name](self.data_path, train=True, download=False,
                                                                     transform=self.train_transform)
        self.test_dataset = DataPreparer.dataset_func[dataset_name](self.data_path, train=False, download=False,
                                                                    transform=self.test_transform)
        self.batch_size = batch_size

        if sampler is None:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank,seed=3)
        else:
            self.train_sampler = sampler(self.train_dataset,world_size,rank) #BatchSampler(sampler=sampler(self.train_dataset,world_size,rank),batch_size=batch_size,drop_last=False)

        self.test_sampler = DistributedSampler(self.test_dataset, num_replicas=world_size, rank=rank)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=(self.train_sampler is None),
                                       sampler=self.train_sampler, num_workers=2, persistent_workers =False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      sampler=self.test_sampler,
                                      num_workers=2, persistent_workers=False)

    def set_epoch(self,epoch):
        if self.train_sampler is not None and hasattr(self.train_sampler,"set_epoch"):
            self.train_sampler.set_epoch(epoch)
