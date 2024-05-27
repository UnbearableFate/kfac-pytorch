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

if __name__ == '__main__':
    timeout = datetime.timedelta(seconds=30)
    dist.init_process_group("gloo", timeout=timeout)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    #model_fn = partial(ResNetForCIFAR10, layers=18)
    #model = MLP()
    model = ResNetForCIFAR10(layers=18)
    general_util.general_main(data_dir= DATA_DIR,log_dir= LOG_DIR ,dataset_name= "CIFAR10",
                              timestamp= timestamp,model=model,model_avg_interval=5,disconnect_ratio=0.2)
    #general_util.general_main(data_dir= DATA_DIR,log_dir= LOG_DIR ,dataset_name= "FashionMNIST",
    #                          timestamp= timestamp, model_func=MLP,model_avg_time_in_one_epoch=21,disconnect_ratio=0.4)
    dist.destroy_process_group()
    print("Done!")