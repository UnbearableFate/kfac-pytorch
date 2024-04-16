from torch.utils.data import Sampler
import numpy as np

class NonIIDSampler(Sampler):
    def __init__(self, dataset, num_clients, num_samples_per_client):
        self.dataset = dataset
        self.num_clients = num_clients
        self.num_samples_per_client = num_samples_per_client
        self.total_samples = num_samples_per_client * num_clients

        # 假设数据集已根据标签排序或至少集中
        self.indices = self.divide_indices()

    def divide_indices(self):
        # 基于标签分配索引
        labels = np.array([label for _, label in self.dataset])
        idxs_per_label = {label: np.where(labels == label)[0] for label in np.unique(labels)}

        client_indices = []
        for i in range(self.num_clients):
            # 每个客户端选择的标签范围可以调整以制造非IID
            chosen_indices = []
            for label in np.random.choice(list(idxs_per_label.keys()), size=2, replace=False):
                chosen_indices.extend(np.random.choice(idxs_per_label[label], size=self.num_samples_per_client // 2, replace=False))
            client_indices.append(np.array(chosen_indices))
        return client_indices

    def __iter__(self):
        for idxs in self.indices:
            np.random.shuffle(idxs)
            yield idxs

    def __len__(self):
        return self.total_samples