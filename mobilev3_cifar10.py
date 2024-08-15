import datetime
import os
import argparse
import torch

import kfac
from my_module.custom_resnet import ResNetForCIFAR10, MLP ,SimpleCNN
from general_util.GeneralManager import GeneralManager
from my_module.mobile_net import CustomMiniMobileNetV3Small, CustomMobileNetV3Small
from my_module.model_split import ModelSplitter

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
if __name__ == '__main__':
    print("Start!")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    parser = argparse.ArgumentParser(description="experiment script")
    parser.add_argument('--timestamp', type=str, default=timestamp)
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"timestamp: {timestamp}")

    model = CustomMobileNetV3Small(num_classes=10)
    #device = torch.device(f"cuda:0")
    device = torch.device(f"cuda:{ompi_world_rank%4}")
    model = model.to(device)
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model, skip_layers=["block.0.0", "block.1.0"])

    data_path = DATA_DIR + str(ompi_world_rank)
    mgr = GeneralManager(data_dir=DATA_DIR, dataset_name="CIFAR10", model=model,
                         sampler_func= None,
                         train_com_method='rpc', interval=1, is_2nd_order=True, epochs=5,device=device,
                         share_file_path=Share_DIR,timestamp=timestamp, log_dir = LOG_DIR ,precondtioner=preconditioner)

    mgr.rpc_train_and_test()
    mgr.close_all()
    print("Done!")
