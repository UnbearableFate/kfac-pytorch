import datetime
import os
import argparse
import torch

import kfac
from my_module.custom_resnet import ResNetForCIFAR10, MLP ,SimpleCNN
from general_util.GeneralManager import GeneralManager
from my_module.model_split import ModelSplitter
import torch.distributed as dist
import logging

logging.basicConfig(level=logging.NOTSET)

gpu = torch.device("cuda:0")
today = datetime.date.today().strftime('%m%d')
pg_share_file = "pg_share"
rpc_share_fie = "rpc_share"

DATA_DIR = ""
LOG_DIR = ""
Share_DIR = ""
if os.path.exists("/home/yu"):
    DATA_DIR = "/home/yu/data"
    LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs/runs"+today
    Share_DIR = "/home/yu/workspace/kfac-pytorch/share_files"
elif os.path.exists("/Users/unbearablefate"):
    DATA_DIR = "/Users/unbearablefate/workspace/data"
    LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs/runs"+today
    Share_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/share_files"
elif os.path.exists("/work/NBB/yu_mingzhe/kfac-pytorch"):
    DATA_DIR = "/work/NBB/yu_mingzhe/data"
    LOG_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/runs/runs"+today
    Share_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/share_files"

if DATA_DIR == "" or LOG_DIR == "" or Share_DIR == "":
    raise RuntimeError("Unknown environment.")

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
if ompi_world_rank == 0:
    logging.basicConfig(level=logging.NOTSET)

if __name__ == '__main__':
    print("Start!")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    parser = argparse.ArgumentParser(description="experiment script")
    parser.add_argument('--timestamp', type=str, default=timestamp)
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"timestamp: {timestamp}")

    timeout = datetime.timedelta(seconds=120)
    dist.init_process_group("gloo", init_method=f"file://{Share_DIR}/pg_share{timestamp}", rank=ompi_world_rank,
                            world_size=ompi_world_size, timeout=timeout)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")

    model = MLP(num_hidden_layers=8,hidden_size=32)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank%4}")
    #device = torch.device(f"cpu")
    model = model.to(device)
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model, skip_layers=["layer.1"], damping= 0.003 ,inv_update_steps=13)
    mgr = GeneralManager(data_dir=DATA_DIR, dataset_name="FashionMNIST", model=model,
                         sampler_func= None,
                         train_com_method='rpc', interval=11, is_2nd_order=True, epochs=1,batch_size=4 ,device=device,
                         share_file_path=Share_DIR,timestamp=timestamp, log_dir = LOG_DIR ,precondtioner=preconditioner)

    mgr.rpc_train_and_test()
    mgr.close_all()
    print("Done!")
