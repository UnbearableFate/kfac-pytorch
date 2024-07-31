import datetime
import os
import argparse
import torch
from my_module.custom_resnet import ResNetForCIFAR10, MLP ,SimpleCNN
from general_util.GeneralManager import GeneralManager
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
    LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs"+today
    Share_DIR = "/home/yu/workspace/kfac-pytorch/share_files"
elif os.path.exists("/Users/unbearablefate"):
    DATA_DIR = "/Users/unbearablefate/workspace/data"
    LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs"+today
    Share_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/share_files"
elif os.path.exists("/work/NBB/yu_mingzhe/kfac-pytorch"):
    DATA_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data"
    LOG_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/runs"+today
    Share_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data/share_files"

if DATA_DIR == "" or LOG_DIR == "" or Share_DIR == "":
    raise RuntimeError("Unknown environment.")

if __name__ == '__main__':
    print("Start!")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    parser = argparse.ArgumentParser(description="experiment script")
    parser.add_argument('--timestamp', type=str, default=timestamp)
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"timestamp: {timestamp}")
    model = SimpleCNN()
    model = ModelSplitter(model, 128)
    mgr = GeneralManager(data_dir=DATA_DIR, dataset_name="CIFAR10", model=model,
                         sampler_func= None,
                         train_com_method='rpc', interval=1, is_2nd_order=True, epochs=40,device='cpu',
                         share_file_path=Share_DIR,timestamp=timestamp, log_dir = LOG_DIR,trainsform_train=SimpleCNN.transform,transform_test=SimpleCNN.transform)

    mgr.rpc_train_and_test(log_dir=LOG_DIR, timestamp=timestamp, experiment_name="test05")
    mgr.close_all()
    print("Done!")