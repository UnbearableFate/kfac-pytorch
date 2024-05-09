
import datetime
import torch.distributed as dist
import general_util
from my_module.custom_resnet import ResNetForCIFAR10
from functools import partial
import argparse

import os
ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', 0))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', 0))
DATA_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data"
LOG_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/runs"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="experiment script")
    parser.add_argument('--disconnect_ratio', type=float, help='disconnect_ratio_float',default=0.2)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    parser.add_argument('--timestamp', type=str, default=timestamp)
    parser.add_argument('--max_sick_iter_ratio', type=float, help='max_sick_iter_ratio',default=0.2)

    args = parser.parse_args()

    timeout = datetime.timedelta(seconds=120)
    dist.init_process_group("nccl", timeout=timeout,init_method='file:///work/NBB/yu_mingzhe/kfac-pytorch/data/sharedfile',
                            world_size=ompi_world_size, rank=ompi_world_rank)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")
    
    timestamp = args.timestamp
    
    model = ResNetForCIFAR10(layers=101)
    
    general_util.general_main(data_dir= DATA_DIR,log_dir= LOG_DIR ,dataset_name= "CIFAR10",
                              timestamp= timestamp,model=model,
                              model_avg_interval=5,
                              disconnect_ratio=args.disconnect_ratio,
                              max_sick_iter_ratio=args.max_sick_iter_ratio)
    
    #general_util.nora_main(data_dir= DATA_DIR,log_dir= LOG_DIR ,dataset_name= "CIFAR10",model=model)
    dist.destroy_process_group()