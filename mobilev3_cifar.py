import datetime
import os
import argparse
import torch

import kfac
from my_module.custom_resnet import ResNetForCIFAR10, MLP ,SimpleCNN
from general_util.GeneralManager import GeneralManager
from my_module.mobile_net import CustomMiniMobileNetV3Small, CustomMobileNetV3Small
from my_module.model_split import ModelSplitter
from torchvision import transforms
import torch.distributed as dist
import logging


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
    model = CustomMobileNetV3Small(num_classes=10)
    device = torch.device(f"cuda:{ompi_world_rank%4}")
    model = model.to(device)
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model, skip_layers=["block.0.0", "block.1.0"],damping=0.007, inv_update_steps=17)

    transform = transforms.Compose([
        transforms.Resize(224),  # 将图像大小调整为224x224
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道的RGB图像
        transforms.ToTensor(),  # 将图像转换为张量，并且将像素值缩放到 [0, 1] 范围内
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 对RGB通道进行标准化
                             std=[0.229, 0.224, 0.225]),
    ])

    data_path = DATA_DIR + str(ompi_world_rank%8)
    mgr = GeneralManager(data_dir=data_path, dataset_name="CIFAR10", model=model,
                         sampler_func= None,
                         train_com_method='rpc', interval=13, is_2nd_order=True, epochs=50, device=device,
                         share_file_path=Share_DIR, timestamp=timestamp, log_dir = LOG_DIR, precondtioner=preconditioner,
                         transform_train=None, transform_test=None)

    mgr.rpc_train_and_test()
    mgr.close_all()
    print("Done!")
