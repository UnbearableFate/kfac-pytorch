import datetime
import os
import argparse
import torch

import kfac
from kfac.enums import ComputeMethod
from my_module.custom_resnet import ResNetForCIFAR10, MLP ,SimpleCNN
from general_util.GeneralManager import GeneralManager
from my_module.model_split import ModelSplitter
from torchvision import transforms
import logging
import torch.distributed as dist
from general_util.consts import DATA_DIR, LOG_DIR, SHARE_FILES_DIR

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

gpu = torch.device("cuda:0")
today = datetime.date.today().strftime('%m%d')
pg_share_file = "pg_share"
rpc_share_fie = "rpc_share"

if DATA_DIR == "" or LOG_DIR == "" or SHARE_FILES_DIR== "":
    raise RuntimeError("Unknown environment.")

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
from mpi4py import MPI
if ompi_world_size == -1 or ompi_world_rank == -1:
    ompi_world_rank = MPI.COMM_WORLD.Get_rank()
    ompi_world_size = MPI.COMM_WORLD.Get_size()

if ompi_world_rank == 0:
    logging.basicConfig(level=logging.NOTSET)

if __name__ == '__main__':
    print(f"Start! at {datetime.datetime.now()}")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    parser = argparse.ArgumentParser(description="experiment script")
    parser.add_argument('--timestamp', type=str, default=timestamp)
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"timestamp: {timestamp}")
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA Devices: {num_devices}")

    timeout = datetime.timedelta(seconds=120)
    dist.init_process_group("gloo", init_method=f"file://{SHARE_FILES_DIR}/pg_share{timestamp}", rank=ompi_world_rank,
                            world_size=ompi_world_size, timeout=timeout)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")

    model = ResNetForCIFAR10(layers=34)
    device = torch.device(f"cuda:0")
    model = model.to(device)
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model, damping=0.007,lr=0.1,train_method='rpc',is_packaged_send=True)

    transform = transforms.Compose([
        transforms.Resize(224),  # 将图像大小调整为224x224
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道的RGB图像
        transforms.ToTensor(),  # 将图像转换为张量，并且将像素值缩放到 [0, 1] 范围内
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 对RGB通道进行标准化
                             std=[0.229, 0.224, 0.225]),
    ])

    mgr = GeneralManager(dataset_name="CIFAR10", model=model,
                         sampler_func= None,
                         train_com_method='rpc',  is_2nd_order=True, epochs=75, device=device,
                         timestamp=timestamp,  precondtioner=preconditioner,
                         transform_train=None, transform_test=None,experiment_name="ring_swift_resnet34",
                         recover=False,batch_size=256)

    mgr.rpc_train_and_test()
    print(f"Done! at {datetime.datetime.now()}")

# cd /work/NBB/yu_mingzhe/kfac-pytorch
# module load openmpi/4.1.6/nvhpc24.5-cuda12.4 
# conda activate py311
# mpirun -n 4 python ./multi_node_resnet_cifar.py

# cd /work/xg24i002/x10041/kfac-pytorch
# psutil scipy

# CFLAGS=-noswitcherror pip install mpi4py
# pip install tensorboard