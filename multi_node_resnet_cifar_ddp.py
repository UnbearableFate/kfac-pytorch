import datetime
import os
import argparse
import torch

import kfac
from my_module.custom_resnet import ResNetForCIFAR10, MLP ,SimpleCNN
from general_util.GeneralManager import GeneralManager
from my_module.model_split import ModelSplitter
from torchvision import transforms
import torch.distributed as dist
from kfac.enums import ComputeMethod
from general_util.consts import DATA_DIR, LOG_DIR, SHARE_FILES_DIR ,ompi_world_size, ompi_world_rank, parse_args

import examples.vision.cifar_resnet as models
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == '__main__':
    args = parse_args()
    timestamp = args.timestamp
    
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA Devices: {num_devices} at rank {ompi_world_rank} at hostname: {os.uname().nodename}")

    timeout = datetime.timedelta(seconds=120)
    dist.init_process_group("nccl", init_method=f"file://{SHARE_FILES_DIR}/pg_share{timestamp}", rank=ompi_world_rank,
                            world_size=ompi_world_size, timeout=timeout)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")

    model = ResNetForCIFAR10(layers=34)
    if num_devices > 1:
        device = torch.device(f"cuda:{ompi_world_rank%num_devices}")
    else:
        device = torch.device("cuda:0")
    model = model.to(device)
    model = DDP(model)
    preconditioner = kfac.preconditioner.KFACPreconditioner(model=model,damping=0.007,factor_update_steps = 15,inv_update_steps=60)

    transform = transforms.Compose([
        transforms.Resize(224),  # 将图像大小调整为224x224
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道的RGB图像
        transforms.ToTensor(),  # 将图像转换为张量，并且将像素值缩放到 [0, 1] 范围内
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 对RGB通道进行标准化
                             std=[0.229, 0.224, 0.225]),
    ])

    mgr = GeneralManager(dataset_name="CIFAR10", model=model,
                         sampler_func= None,
                         train_com_method='ddp', is_2nd_order=True, epochs=75, device=device,
                         timestamp=timestamp,  precondtioner=preconditioner,
                         transform_train=None, transform_test=None,experiment_name="ddp_kfac_resnet34",
                         recover=False, batch_size=256)

    mgr.train_and_test()
    print(f"Done! at {datetime.datetime.now()}")
    exit(0)
