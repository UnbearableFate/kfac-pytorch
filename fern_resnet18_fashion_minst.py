import datetime
import math
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import logging
import kfac
import kfac.mischief as mischief
from my_module.custom_resnet import ResNetForCIFAR10, MLP
from general_util.data_preparation import DataPreparer, SimpleNonIIDSampler
import general_util
from functools import partial
gpu = torch.device("cuda:0")
DATA_DIR = "/home/yu/data"
LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs"

def main():
    # Set up DDP environment
    timeout = datetime.timedelta(seconds=30)
    dist.init_process_group('gloo', timeout=timeout)
    if not dist.is_initialized():
        return
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        logging.basicConfig(level=logging.NOTSET)

    data_manager = DataPreparer(data_path_root=DATA_DIR, dataset_name="FashionMNIST", world_size=world_size, rank=rank,
                                sampler=None, batch_size=64)
    batch_size = 64
    sick_iter_ratio = 0.8
    if rank == 0:
        print(f"iteration in loop {int(len(data_manager.train_dataset) / batch_size /world_size)}")
    max_disconnect_iter = int(len(data_manager.train_dataset) / batch_size / world_size * sick_iter_ratio)
    disconnect_ratio = 0.2
    max_disconnected_node_num = world_size // 2
    mischief.mischief_init(world_size=world_size, possible_disconnect_node=None,
                           max_disconnect_iter=max_disconnect_iter, disconnect_ratio=disconnect_ratio,
                            max_disconnected_node_num=max_disconnected_node_num,
                           ddp_trigger=True, factor_comm_trigger=True, inverse_comm_trigger=True)

    # Define the model, loss function, and optimizer
    model = MLP().to(gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model, update_factors_in_hook=False)

    experiment_name_detail = f"mdn{max_disconnected_node_num}_dr{disconnect_ratio}_mdi{max_disconnect_iter}_ws{world_size}"
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    writer = SummaryWriter(
        log_dir=os.path.join(LOG_DIR, f"fashion_minist_mlp8/240430/{experiment_name_detail}_{timestamp}/{dist.get_rank()}"))

    mgr = general_util.GeneralManager(model=model, data_manager=data_manager, loss_func=criterion, optimizer=optimizer,
                                      preconditioner=preconditioner, epochs=50, world_size=world_size, rank=rank, device=gpu,
                                      writer=writer)
    mgr.train_and_test()

    if rank == 0:
        mischief.print_node_status()
    

    # Testing function

if __name__ == '__main__':
    timeout = datetime.timedelta(seconds=30)
    dist.init_process_group("gloo", timeout=timeout)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    model_fn = partial(ResNetForCIFAR10, layers=18)
    general_util.general_main(data_dir= DATA_DIR,log_dir= LOG_DIR ,dataset_name= "CIFAR10",
                              timestamp= timestamp,model_func=model_fn,disconnect_ratio=0.2,device='cpu')
    general_util.general_main(data_dir= DATA_DIR,log_dir= LOG_DIR ,dataset_name= "CIFAR10",
                              timestamp= timestamp, ModelType=model_fn,disconnect_ratio=0.4,device='cpu')
    dist.destroy_process_group()
    print("Done!")