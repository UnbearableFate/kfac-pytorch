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
from general_util.GeneralManager import GeneralManager
from functools import partial
gpu = torch.device("cuda:0")
DATA_DIR = "/home/yu/data"
LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs3"

if __name__ == '__main__':
    timeout = datetime.timedelta(seconds=30)
    dist.init_process_group("gloo", timeout=timeout)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    model = MLP()
    mgr = GeneralManager(data_dir=DATA_DIR, dataset_name="FashionMNIST", model=model,
                         sampler_func= SimpleNonIIDSampler,
                         is_ddp=False, interval=10, is_2nd_order=True, epochs=2)
    mgr.init_mischief(disconnect_ratio=0.2, max_sick_iter_ratio=0.2, max_disconnected_node_num=2)
    mgr.train_and_test(log_dir=LOG_DIR, timestamp=timestamp, experiment_name="ddp_test")
    dist.destroy_process_group()
    print("Done!")