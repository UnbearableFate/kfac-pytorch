import datetime
import torch
import torch.distributed as dist
from my_module.custom_resnet import ResNetForCIFAR10, MLP
from general_util.GeneralManager import GeneralManager
gpu = torch.device("cuda:0")
import debugpy
import os
import sys
#DATA_DIR = "/Users/unbearablefate/workspace/data"
#LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs2"
DATA_DIR = "/home/yu/data"
LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs9"
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12345'

if __name__ == '__main__':
    timeout = datetime.timedelta(seconds=30)
    dist.init_process_group("gloo", timeout=timeout,init_method='file:///home/yu/workspace/kfac-pytorch/dist_file',
                            world_size=4, rank=int(sys.argv[1]))
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    model = MLP(num_hidden_layers=3)
    mgr = GeneralManager(data_dir=DATA_DIR, dataset_name="FashionMNIST", model=model,
                         sampler_func= None,
                         is_ddp=False, interval=3, is_2nd_order=True, epochs=7,device='cpu')
    #mgr.init_mischief(disconnect_ratio=0.2, max_sick_iter_ratio=0.2, max_disconnected_node_num=2)
    mgr.train_and_test(log_dir=LOG_DIR, timestamp=timestamp, experiment_name="ddp_test")
    dist.destroy_process_group()
    print("Done!")