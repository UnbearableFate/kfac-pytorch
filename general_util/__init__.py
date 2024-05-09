import datetime
import torch
from tqdm import tqdm
import kfac.mischief as mischief
from general_util.data_preparation import DataPreparer
import torch.distributed as dist
import logging
import kfac
import kfac.mischief as mischief
from my_module.custom_resnet import CustomResNet18, MLP
from general_util.data_preparation import DataPreparer, SimpleNonIIDSampler
from torch.utils.tensorboard import SummaryWriter
from general_util.GeneralManager import GeneralManager
import torch.nn as nn
import os
from torch.nn.parallel import DistributedDataParallel as DDP


def nora_main(data_dir,log_dir,dataset_name,model,device=torch.device("cuda:0")):
    batch_size=64
    epochs=100
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    mischief.mischief_init(world_size=world_size, possible_disconnect_node=[],
                           max_disconnect_iter=2, disconnect_ratio=0,
                            max_disconnected_node_num=0,
                           ddp_trigger=False, factor_comm_trigger=False, inverse_comm_trigger=False)

    data_manager = DataPreparer(data_path_root=data_dir, dataset_name=dataset_name, world_size=world_size, rank=rank,
                                sampler=None, batch_size=batch_size)

    # Define the model, loss function, and optimizer
    model_name =  type(model).__name__
    if hasattr(model, "model_name"):
        model_name = model.model_name

    model = model.to(device)
    model = DDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model)
    writer_name = f"{dataset_name}_{model_name}/normal/{dist.get_rank()}"
    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, writer_name))
    mgr = GeneralManager(model=model, data_manager=data_manager, loss_func=criterion, optimizer=optimizer,
                                      preconditioner=preconditioner, epochs=epochs, world_size=world_size, rank=rank, device=device,
                                      writer=writer)
    mgr.train_and_test_with_normal_ddp()
    writer.close()

def general_main(data_dir,log_dir,dataset_name,timestamp,model,device=torch.device("cuda:0"),
                  model_avg_interval=10,
                  max_sick_iter_ratio=0.2, disconnect_ratio = 0.2,
                  possible_disconnect_node = None ,max_disconnected_node_num = 2,
                  ):
    batch_size=64
    epochs=100

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        logging.basicConfig(level=logging.NOTSET)

    data_manager = DataPreparer(data_path_root=data_dir, dataset_name=dataset_name, world_size=world_size, rank=rank,
                                sampler=None, batch_size=batch_size)
    if rank == 0:
        print(f"iteration in loop {int(len(data_manager.train_dataset) / batch_size /world_size)}")
    max_disconnect_iter = int(len(data_manager.train_dataset) / batch_size / world_size * max_sick_iter_ratio)
    
    mischief.mischief_init(world_size=world_size, possible_disconnect_node=possible_disconnect_node,
                           max_disconnect_iter=max_disconnect_iter, disconnect_ratio=disconnect_ratio,
                            max_disconnected_node_num=max_disconnected_node_num,
                           ddp_trigger=True, factor_comm_trigger=True, inverse_comm_trigger=True)

    # Define the model, loss function, and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model)

    model_name =  type(model).__name__
    if hasattr(model, "model_name"):
        model_name = model.model_name
    experiment_name_detail = f"mdn{max_disconnected_node_num}_dr{disconnect_ratio}_mdi{max_disconnect_iter}_ws{world_size}"
    writer_name = f"{dataset_name}_{model_name}/{timestamp}/{experiment_name_detail}/{dist.get_rank()}"
    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, writer_name))

    mgr = GeneralManager(model=model, data_manager=data_manager, loss_func=criterion, optimizer=optimizer,
                                      preconditioner=preconditioner, epochs=epochs, world_size=world_size, rank=rank, device=device,
                                      writer=writer,interval=model_avg_interval)
    #mgr.train_and_test()
    mgr.train_and_test_async()
    if rank == 0:
        mischief.print_node_status()

    writer.close()
