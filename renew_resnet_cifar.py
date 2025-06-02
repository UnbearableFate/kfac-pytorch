import datetime
import os
import torch
from torch.utils import collect_env
from my_module.custom_resnet import ResNetForCIFAR10, MLP
from general_util.GeneralManager import GeneralManager
import torch.distributed as dist
from general_util.consts import DATA_DIR, LOG_DIR, SHARE_FILES_DIR ,ompi_world_size, ompi_world_rank, parse_args
from torch.distributed import rpc
import examples.vision.cifar_resnet as models
from torch.nn.parallel import DistributedDataParallel as DDP
from my_module.my_swin import SwinTransformer, SwinTransformerBlockV2
import time

import logging

if ompi_world_rank == 0:
    logging.basicConfig(level=logging.NOTSET)

if __name__ == '__main__':
    args = parse_args()
    timestamp = args.timestamp
    
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA Devices: {num_devices} at rank {ompi_world_rank} at hostname: {os.uname().nodename}")

    timeout = datetime.timedelta(seconds=60)

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

    if args.model == 'resnet':
        model = ResNetForCIFAR10(layers=args.layers)
        device = torch.device(f'cuda:0')
    elif args.model == 'mlp':
        model = MLP(num_hidden_layers=args.layers, hidden_size=256)
        device = torch.device(f'cuda:0')
    elif args.model == 'swin':
        model = SwinTransformer(
            patch_size=[2, 2],            # 更小的patch_size以适应32x32输入
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[4, 4],           # 缩小window_size以适应更小的图像尺寸
            num_classes=10,
            stochastic_depth_prob=0.2,
        )
        device = torch.device(f'cuda:0')

    model = model.to(device)
    
    if args.train_com_method == 'ddp':
        model = DDP(model)
    start_time = time.time() 
    mgr = GeneralManager(model=model,device=device, args=args)
    dist.barrier()
    if args.train_com_method == 'ddp':
        mgr.train_and_test()
    else:
        mgr.rpc_train_and_test()

    timelog = f"total time: {time.time()-start_time}"
    if args.train_com_method == 'rpc':
        mgr.rpc_communicator.debug_print(timelog)
    print(timelog)
    dist.barrier()
    if args.train_com_method == 'rpc':
        rpc.shutdown()
    dist.destroy_process_group()
    print(f"Done! at {datetime.datetime.now()}")
    exit(0)
