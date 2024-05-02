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

def general_main(data_dir,log_dir,dataset_name,timestamp,device=torch.device("cuda:0"),
                  max_sick_iter_ratio=0.2, disconnect_ratio = 0.2,
                  possible_disconnect_node = None ,max_disconnected_node_num = 2,
                  model_func = MLP):
    # Set up DDP environment
    
    batch_size=64
    epochs=1

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
    model = model_func.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model, update_factors_in_hook=False)

    experiment_name_detail = f"mdn{max_disconnected_node_num}_dr{disconnect_ratio}_mdi{max_disconnect_iter}_ws{world_size}"
    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, f"{dataset_name}_{model.__name__}/{timestamp}/{experiment_name_detail}/{dist.get_rank()}"))

    mgr = GeneralManager(model=model, data_manager=data_manager, loss_func=criterion, optimizer=optimizer,
                                      preconditioner=preconditioner, epochs=epochs, world_size=world_size, rank=rank, device=device,
                                      writer=writer)
    mgr.train_and_test()

    if rank == 0:
        mischief.print_node_status()
    
    writer.close()
