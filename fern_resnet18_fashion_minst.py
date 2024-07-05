import datetime
import torch
import torch.distributed as dist
from my_module.custom_resnet import ResNetForCIFAR10, MLP
from general_util.GeneralManager import GeneralManager
from my_module.model_split import ModelSplitter
gpu = torch.device("cuda:0")
import os
#DATA_DIR = "/Users/unbearablefate/workspace/data"
#LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs0627"
DATA_DIR = "/home/yu/data"
LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs0705"
import logging

if __name__ == '__main__':
    #timeout = datetime.timedelta(seconds=30)
    dist.init_process_group("gloo")
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    model = MLP(num_hidden_layers=3,hidden_size=64)
    model = ModelSplitter(model, 64)
    mgr = GeneralManager(data_dir=DATA_DIR, dataset_name="FashionMNIST", model=model,
                         sampler_func= None,
                         train_com_method='ddp', interval=1, is_2nd_order=True, epochs=100,device='cpu')
    #mgr.init_mischief(disconnect_ratio=0.2, max_sick_iter_ratio=0.2, max_disconnected_node_num=2)
    mgr.train_and_test(log_dir=LOG_DIR, timestamp=timestamp, experiment_name="test01")
    dist.destroy_process_group()
    print("Done!")