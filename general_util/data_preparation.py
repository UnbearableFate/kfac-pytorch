import os.path
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, BatchSampler

from torch.utils.data import Sampler
import numpy as np
from enum import Enum

import math
import random
from torch.utils.data import Sampler

class NonIidSampler(Sampler):
    """
    根据 degree_noniid 参数生成非IID采样序列，并在每个 epoch 开始时打乱顺序。
    
    数据分区逻辑：
      1. 根据 dataset.targets 将所有样本按标签分组。
      2. 按每个 worker 分得样本数计算非IID部分（ceil(total * degree_noniid)）和 IID 部分。
      3. 对每个标签按照 degree_noniid 划分出非IID数据，其余作为 IID 数据。
      4. 通过轮转方式为每个 worker 分配非IID数据，再为 IID 数据分配不重叠的片段。
      5. 最终合并非IID和 IID 部分构成当前 worker 的样本索引。
    
    set_epoch 方法可以在每个 epoch 开始时更新随机种子，从而打乱当前 worker 内部的顺序。
    """
    def __init__(self, dataset, world_size, rank, degree_noniid=0.7, seed=1234):
        """
        参数：
         - dataset: 训练数据集，要求具有 targets 属性（记录每个样本的标签）
         - num_workers: 总共的 worker 数量
         - rank: 当前 worker 的编号（0-indexed）
         - degree_noniid: 非IID程度，取值范围 [0, 1]
         - seed: 随机种子，用于数据划分和后续打乱
        """
        self.dataset = dataset
        self.num_workers = world_size
        self.rank = rank
        self.degree_noniid = max(0, min(1, degree_noniid))
        self.seed = seed
        self.epoch = 0  # 记录当前 epoch
        self.base_indices = self._generate_indices()
        
    def _generate_indices(self):
        num_data = len(self.dataset)
        # 按 worker 平均分配样本数量
        partition_sizes = [1.0 / self.num_workers for _ in range(self.num_workers)]
        total_worker_sizes = [int(p * num_data) for p in partition_sizes]
        total_assigned = sum(total_worker_sizes)
        rem = num_data - total_assigned
        for i in range(rem):
            total_worker_sizes[i] += 1
        
        worker_total = total_worker_sizes[self.rank]
        non_iid_count = math.ceil(worker_total * self.degree_noniid)
        iid_count = worker_total - non_iid_count
        
        # 1. 按标签分组
        label_list = self.dataset.targets
        label_idx_dict = {}
        for idx, label in enumerate(label_list):
            label_idx_dict.setdefault(label, []).append(idx)
        
        labels = list(label_idx_dict.keys())
        num_labels = len(labels)
        
        # 2. 根据每个标签的样本数计算用于非IID分配的数量
        total_label_niids = {}
        for label in labels:
            total_label_niids[label] = int(len(label_idx_dict[label]) * self.degree_noniid)
        
        # 调整各标签的非IID数量，使得所有 worker 需要的非IID样本总数匹配
        non_iid_sum = sum(math.ceil(total_worker_sizes[w] * self.degree_noniid) for w in range(self.num_workers))
        current_total_niid = sum(total_label_niids[label] for label in labels)
        rem_adjust = non_iid_sum - current_total_niid
        i = 0
        while rem_adjust != 0 and i < len(labels):
            label = labels[i]
            if rem_adjust > 0:
                total_label_niids[label] += 1
                rem_adjust -= 1
            elif rem_adjust < 0 and total_label_niids[label] > 0:
                total_label_niids[label] -= 1
                rem_adjust += 1
            i += 1
            if i >= len(labels) and rem_adjust != 0:
                i = 0
        
        # 3. 对每个标签 shuffle，然后分割成非IID部分和 IID 池
        rng = random.Random(self.seed)
        for label in labels:
            rng.shuffle(label_idx_dict[label])
        non_iid_pool = {}
        iid_pool = []
        for label in labels:
            niid = total_label_niids[label]
            non_iid_pool[label] = label_idx_dict[label][:niid]
            iid_pool.extend(label_idx_dict[label][niid:])
        
        # 4. 轮转方式为各 worker 分配非IID数据
        current_non_iid = []
        curr_label_idx = 0
        for w in range(self.num_workers):
            worker_non_iid = math.ceil(total_worker_sizes[w] * self.degree_noniid)
            assigned = []
            to_fill = worker_non_iid
            while to_fill > 0:
                current_label = labels[curr_label_idx % num_labels]
                available = len(non_iid_pool[current_label])
                take_num = min(to_fill, available)
                if take_num > 0:
                    # 从列表尾部取样，与原逻辑一致
                    assigned.extend(non_iid_pool[current_label][-take_num:])
                    non_iid_pool[current_label] = non_iid_pool[current_label][:-take_num]
                to_fill -= take_num
                curr_label_idx += 1
            if w == self.rank:
                current_non_iid = assigned
                break
        
        # 5. IID 部分：先随机打乱后分段分配，每个 worker 获得不重复的片段
        rng.shuffle(iid_pool)
        iid_start = 0
        for w in range(self.rank):
            worker_iid = total_worker_sizes[w] - math.ceil(total_worker_sizes[w] * self.degree_noniid)
            iid_start += worker_iid
        current_iid = iid_pool[iid_start: iid_start + iid_count]
        
        # 合并非IID和 IID 部分，构成最终索引列表
        final_indices = current_non_iid + current_iid
        return final_indices
    
    def set_epoch(self, epoch):
        """
        在每个 epoch 开始时调用，更新 epoch 后在 __iter__ 中用新的随机种子打乱索引顺序。
        """
        self.epoch = epoch
        
    def __iter__(self):
        # 每个 epoch 用 seed+epoch 生成新的随机顺序打乱 base_indices
        indices = self.base_indices.copy()
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return len(self.base_indices)


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

    def __init__(self, data_path_root, dataset_name, world_size, rank, batch_size=64, sampler=None, train_transform =None,test_transform=None,train_com_method='ddp'):
        self.data_path = os.path.join(data_path_root, dataset_name.lower())
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
            if train_com_method == 'rpc':
                self.train_sampler = None #DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank,seed=7) #None
            else:
                self.train_sampler = None #DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank,seed=7)
        else:
            self.train_sampler = sampler(dataset=self.train_dataset,world_size= world_size,rank = rank) #BatchSampler(sampler=sampler(self.train_dataset,world_size,rank),batch_size=batch_size,drop_last=False)

        if train_com_method == 'ddp':
            self.test_sampler = DistributedSampler(self.test_dataset, num_replicas=world_size, rank=rank)
        else:
            self.test_sampler = None
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=(self.train_sampler is None),
                                       sampler=self.train_sampler, num_workers=2, persistent_workers =True,pin_memory=True, prefetch_factor=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      sampler=self.test_sampler,
                                      num_workers=2, persistent_workers=True,pin_memory=True)

    def set_epoch(self,epoch):
        if self.train_sampler is not None and hasattr(self.train_sampler,"set_epoch"):
            self.train_sampler.set_epoch(epoch)
