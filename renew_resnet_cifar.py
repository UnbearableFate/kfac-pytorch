import datetime
import os
import torch
from torch.utils import collect_env
from my_module.custom_resnet import ResNetForCIFAR10, MLP ,SimpleCNN
from general_util.GeneralManager import GeneralManager
import torch.distributed as dist
from general_util.consts import DATA_DIR, LOG_DIR, SHARE_FILES_DIR ,ompi_world_size, ompi_world_rank, parse_args

import examples.vision.cifar_resnet as models
from torch.nn.parallel import DistributedDataParallel as DDP

import logging

if ompi_world_rank == 0:
    logging.basicConfig(level=logging.NOTSET)

if __name__ == '__main__':
    args = parse_args()
    timestamp = args.timestamp
    
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA Devices: {num_devices} at rank {ompi_world_rank} at hostname: {os.uname().nodename}")

    timeout = datetime.timedelta(seconds=120)

    if args.train_com_method == 'ddp':
        backend = 'nccl'
    else:
        backend = 'gloo'

    dist.init_process_group(backend, init_method=f"file://{SHARE_FILES_DIR}/pg_share{timestamp}", rank=ompi_world_rank,
                            world_size=ompi_world_size, timeout=timeout)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")

    args.verbose = dist.get_rank() == 0

    if args.verbose:
        print('Collecting env info...')
        print(collect_env.get_pretty_env_info())
        print()

    for r in range(dist.get_world_size()):
        if r == dist.get_rank():
            print(
                f'Global rank {dist.get_rank()} initialized: '
                f'local_rank = {args.local_rank}, '
                f'world_size = {dist.get_world_size()}',
            )
        dist.barrier()

    device = torch.device(f'cuda:0')
    model = models.get_model(args.model)
    model = model.to(device)
    
    if args.train_com_method == 'ddp':
        model = DDP(model)

    mgr = GeneralManager(model=model,device=device, args=args)
    mgr.train_and_test()
    print(f"Done! at {datetime.datetime.now()}")
    exit(0)
