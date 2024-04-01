import datetime
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
#import kfac
from kfac.mischief2 import Mischief, MischiefHelper , add_hook_to_model

epochs = 100
batch_size = 128
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
    timeout = datetime.timedelta(seconds=20)
    dist.init_process_group('nccl',timeout=timeout)
    #torch.set_default_device("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not dist.is_initialized():
        return
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        # set logging level NOTSET to enable all logging
        logging.basicConfig(level=logging.NOTSET)

    # Load the FashionMNIST dataset
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST('/work/NBB/yu_mingzhe/kfac-pytorch/data/FashionMNIST', train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST('/work/NBB/yu_mingzhe/kfac-pytorch/data/FashionMNIST', train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(train_dataset,num_replicas=dist.get_world_size(),rank=dist.get_rank())
    test_sampler = DistributedSampler(test_dataset,num_replicas=dist.get_world_size(),rank=dist.get_rank())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                                               sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler,
                                              num_workers=4)

    # Define the model, loss function, and optimizer
    
    Mischief.DDP_TRIGGER = True
    Mischief.WORLD_SIZE = dist.get_world_size()
    Mischief.DISCONNECT_RATIO = 0.5
    Mischief.MAX_DISCONNECTED_NODE_NUM = 2
    Mischief.MAX_DISCONNECT_ITER = 5
    MischiefHelper.contruct_node_status([1,2,3])
    
    model = MLP().to(device)
    model = DDP(model)
    add_hook_to_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    preconditioner = None #kfac.preconditioner.KFACPreconditioner(model)
    dist.barrier()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    writer = SummaryWriter(log_dir=f"/work/NBB/yu_mingzhe/experiments/runs3/fashion_mnist_experiment_sgd_sick_{timestamp}/{dist.get_rank()}")
    for epoch in range(epochs):
        train(model,train_loader,train_sampler,criterion, optimizer,preconditioner,epoch,writer)
        if epoch+1 % 10 == 0:
            node_list = set(range(world_size))
            node_list -= Mischief.DISCONNECTING_NODES
            new_pg = dist.new_group(list(node_list))
            for param in model.parameters():
            # 使用all_reduce来计算所有节点的参数和
                print(f"epoch {epoch} ,allreduce : {param.names}")
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM,group=new_pg)
            # 然后平均化
                param.data /= dist.get_world_size(group=new_pg)
            dist.destroy_process_group(group=new_pg)
        test(model,test_loader,epoch,writer)
    dist.destroy_process_group()

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
            data = data.cuda()
            target = target.cuda()
            MischiefHelper.update_iter()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            #preconditioner.step()
            optimizer.step()
            t.update()
        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), epoch)
    
    # Testing function
def test(model, test_loader,epoch,writer):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = correct / total
    writer.add_scalar('Precision/test', accuracy, epoch)

if __name__ == '__main__':
    main()
    print("Done!")