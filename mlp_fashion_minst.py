import datetime
import os
import argparse
import torch

import kfac
from my_module.custom_resnet import MLP
from general_util.GeneralManager import GeneralManager
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
check_point_path = ""
if os.path.exists("/home/yu"):
    DATA_DIR = "/home/yu/data"
    LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs/runs"+today
    Share_DIR = "/home/yu/workspace/kfac-pytorch/share_files"
    check_point_path = "/home/yu/workspace/kfac-pytorch/checkpoints"
elif os.path.exists("/Users/unbearablefate"):
    DATA_DIR = "/Users/unbearablefate/workspace/data"
    LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs/runs"+today
    Share_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/share_files"
    check_point_path = "/Users/unbearablefate/workspace/kfac-pytorch/checkpoints"
elif os.path.exists("/work/NBB/yu_mingzhe/kfac-pytorch"):
    DATA_DIR = "/work/NBB/yu_mingzhe/data"
    LOG_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/runs/runs"+today
    Share_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/share_files"
    check_point_path = "/work/NBB/yu_mingzhe/kfac-pytorch/checkpoints"

if DATA_DIR == "" or LOG_DIR == "" or Share_DIR == "":
    raise RuntimeError("Unknown environment.")

from mpi4py import MPI
ompi_world_size = MPI.COMM_WORLD.Get_size()
ompi_world_rank = MPI.COMM_WORLD.Get_rank()
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

    model = MLP(num_hidden_layers=4,hidden_size=32)
    rank = dist.get_rank()
    #device = torch.device(f"cuda:{rank%4}")
    device = torch.device(f"cpu")
    model = model.to(device)
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model, skip_layers=["layer.1"], damping= 0.003)
    mgr = GeneralManager(experiment_name="mlp_mnist",dataset_name="FashionMNIST", model=model,
                         train_com_method='rpc', is_2nd_order=True, epochs=8,batch_size=32,device=device,
                         timestamp=timestamp,precondtioner=preconditioner)

    mgr.rpc_train_and_test()
    mgr.close_all()
    print("Done!")
