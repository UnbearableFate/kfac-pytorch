import math
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import kfac

import kfac.mischief as mischief

epochs = 1
batch_size = 32
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

def main():
    # Set up DDP environment
    dist.init_process_group('gloo')
    if not dist.is_initialized():
        return
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mischief.mischief_init(world_size=world_size,possible_disconnect_node=[0,1,2,3],
                           max_disconnect_iter=3,disconnect_ratio=0.2,max_disconnected_node_num=3,
                           ddp_trigger=True, factor_comm_trigger=True, inverse_comm_trigger=True)
    if rank == 0:
        # set logging level NOTSET to enable all logging
        logging.basicConfig(level=logging.NOTSET)

    # Load the FashionMNIST dataset
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST('~/.torch/datasets/FashionMNIST', train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST('~/.torch/datasets/FashionMNIST', train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(train_dataset,num_replicas=dist.get_world_size(),rank=dist.get_rank())
    test_sampler = DistributedSampler(test_dataset,num_replicas=dist.get_world_size(),rank=dist.get_rank())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                                               sampler=train_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler,
                                              num_workers=2)

    # Define the model, loss function, and optimizer
    model = MLP()
    model = DDP(model)
    mischief.add_hook_to_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    preconditioner = kfac.preconditioner.KFACPreconditioner(model)
    dist.barrier()
    writer = SummaryWriter(log_dir=f"runs/fashion_mnist_experiment/{dist.get_rank()}")
    for epoch in range(epochs):
        train(model,train_loader,train_sampler,criterion, optimizer,preconditioner,epoch,writer)

def train(model, train_loader,train_sampler, criterion, optimizer ,preconditioner,epoch,writer):
    model.train()
    train_sampler.set_epoch(epoch)
    with tqdm(
        total=math.ceil(len(train_loader)),
        bar_format='{l_bar}{bar:6}{r_bar}',
        desc=f'Epoch {epoch:3d}/{epochs:3d}',
        disable= (dist.get_rank() != 0)
    ) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            mischief.update_iter()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            preconditioner.step()
            optimizer.step()
            if batch_idx % 10 == 0:
                mischief.print_node_status()
            t.update()

    # Testing function
def test(model, test_loader, criterion, epoch,writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # save loss/epoch into tensorboard
    test_loss /= len(test_loader.dataset)
    if dist.get_rank() == 0:
        writer.add_scalar('Loss/test', test_loss, epoch)

if __name__ == '__main__':
    main()
    dist.barrier()
    dist.destroy_process_group()
    print("Done!")