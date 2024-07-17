import datetime
import os

import torch
import torch.distributed as dist
from my_module.custom_resnet import ResNetForCIFAR10, MLP
from general_util.GeneralManager import GeneralManager
from my_module.model_split import ModelSplitter
ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', 0))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', 0))

gpu = torch.device("cuda:0")
today = datetime.date.today().strftime('%m%d')
pg_share_file = "pg_share"
rpc_share_fie = "rpc_share"

if os.path.exists("/home/yu"):
    DATA_DIR = "/home/yu/data"
    LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs"+today
else:
    DATA_DIR = "/Users/unbearablefate/workspace/data"
    LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs"+today

if __name__ == '__main__':
    #timeout = datetime.timedelta(seconds=30)
    if ompi_world_size <= 0:
        raise RuntimeError("Unable to initialize process group.")

    dist.init_process_group("gloo", init_method=f"file://{DATA_DIR}/{pg_share_file}",
                            rank=ompi_world_rank, world_size=ompi_world_size)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    model = MLP(num_hidden_layers=3,hidden_size=64)
    model = ModelSplitter(model, 64)
    mgr = GeneralManager(data_dir=DATA_DIR, dataset_name="FashionMNIST", model=model,
                         sampler_func= None,
                         train_com_method='rpc', interval=1, is_2nd_order=True, epochs=50,device='cpu')
    #mgr.init_mischief(disconnect_ratio=0.2, max_sick_iter_ratio=0.2, max_disconnected_node_num=2)
    mgr.train_and_test(log_dir=LOG_DIR, timestamp=timestamp, experiment_name="test01")
    dist.destroy_process_group()
    print("Done!")