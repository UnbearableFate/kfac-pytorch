import os
import datetime
import argparse
import torch
import logging

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))

from mpi4py import MPI
if ompi_world_size == -1 or ompi_world_rank == -1:
    ompi_world_rank = MPI.COMM_WORLD.Get_rank()
    ompi_world_size = MPI.COMM_WORLD.Get_size()

if ompi_world_rank == 0:
    logging.basicConfig(level=logging.NOTSET)

DATA_DIR = ""
LOG_DIR = ""
SHARE_FILES_DIR = ""
CHECK_POINT_PATH = ""
SHARED_MODEL_PATH = ""
today = datetime.date.today().strftime('%m%d')

pg_share_file = "pg_share"
rpc_share_fie = "rpc_share"

if os.path.exists("/home/yu"):
    DATA_DIR = "/home/yu/data"
    LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs/runs"+today
    SHARE_FILES_DIR = "/home/yu/workspace/kfac-pytorch/share_files"
    CHECK_POINT_PATH = "/home/yu/workspace/kfac-pytorch/checkpoints"
elif os.path.exists("/Users/unbearablefate"):
    DATA_DIR = "/Users/unbearablefate/workspace/data"
    LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs/runs"+today
    SHARE_FILES_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/share_files"
    CHECK_POINT_PATH = "/Users/unbearablefate/workspace/kfac-pytorch/checkpoints"
elif os.path.exists("/work/NBB/yu_mingzhe/kfac-pytorch"):
    DATA_DIR = "/work/NBB/yu_mingzhe/data"
    LOG_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/runs/runs"+today
    SHARE_FILES_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/share_files"
    CHECK_POINT_PATH = "/work/NBB/yu_mingzhe/kfac-pytorch/checkpoints"
    SHARED_MODEL_ROOT = CHECK_POINT_PATH+"/shared_models"
elif os.path.exists("/work/xg24i002/x10041/kfac-pytorch"):
    DATA_DIR = "/work/xg24i002/x10041/data"
    LOG_DIR = "/work/xg24i002/x10041/kfac-pytorch/runs/runs"+today
    SHARE_FILES_DIR = "/work/xg24i002/x10041/kfac-pytorch/share_files"
    CHECK_POINT_PATH = "/work/xg24i002/x10041/kfac-pytorch/checkpoints"
    SHARED_MODEL_ROOT = CHECK_POINT_PATH+"/shared_models"

delay_list_dict = [
    [0.0006, 0.0071, 0.0046, 0.0043, 0.0119, 0.0049, 0.0121, 0.0097, 0.0029, 0.0238, 0.0069, 0.0062, 0.0008, 0.0148, 0.0114, 0.0098],
    [0.0009, 0.0031, 0.0159, 0.0018, 0.015, 0.0603, 0.0068, 0.0025, 0.0033, 0.0138, 0.0321, 0.0184, 0.0112, 0.0149, 0.0013, 0.0605],
    [0.0016, 0.0311, 0.0176, 0.0288, 0.0139, 0.0399, 0.016, 0.0082, 0.1098, 0.0273, 0.0048, 0.0105, 0.0076, 0.0638, 0.0028, 0.0155],
    [0.0345, 0.0063, 0.0216, 0.1123, 0.0504, 0.0433, 0.0113, 0.0069, 0.0411, 0.0138, 0.0422, 0.0527, 0.0197, 0.0156, 0.0114, 0.0788],
    [0.004, 0.0055, 0.0506, 0.1451, 0.0862, 0.0365, 0.0095, 0.046, 0.116, 0.0207, 0.0056, 0.0395, 0.0476, 0.0403, 0.0086, 0.0179]
]

slowness_threshold = 100 # 正常节点阈值
extreme_threshold = 500 # 极端节点阈值
relative_lag_threshold = 0.08
extreme_relative_lag_threshold = 0.3 # 极端相对滞后阈值
max_election_period = 20

def parse_args() -> argparse.Namespace:
    """Get cmd line args."""
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/tmp/cifar10',
        metavar='D',
        help='directory to download cifar10 dataset to',
    )
    parser.add_argument(
        '--log-dir',
        default='./logs/torch_cifar10',
        help='TensorBoard/checkpoint directory',
    )
    parser.add_argument(
        '--checkpoint-format',
        default='checkpoint_{epoch}.pth.tar',
        help='checkpoint file format',
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        metavar='S',
        help='random seed (default: 42)',
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=False,
        help='use torch.cuda.amp for fp16 training (default: false)',
    )

    # Training settings
    parser.add_argument(
        '--model',
        type=str,
        default='resnet',
        help='ResNet model',
    )

    parser.add_argument(
        '--layers',
        type=int,
        default=34,
        help='number of layers in ResNet (default: 18)',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        metavar='N',
        help='input batch size for training (default: 128)',
    )
    parser.add_argument(
        '--val-batch-size',
        type=int,
        default=128,
        help='input batch size for validation (default: 128)',
    )
    parser.add_argument(
        '--batches-per-allreduce',
        type=int,
        default=1,
        help='number of batches processed locally before '
        'executing allreduce across workers; it multiplies '
        'total batch size.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar='N',
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.1,
        metavar='LR',
        help='base learning rate (default: 0.1)',
    )
    parser.add_argument(
        '--lr-decay',
        nargs='+',
        type=int,
        default=[35, 75, 90],
        help='epoch intervals to decay lr (default: [35, 75, 90])',
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=5,
        metavar='WE',
        help='number of warmup epochs (default: 5)',
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='SGD momentum (default: 0.9)',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        metavar='W',
        help='SGD weight decay (default: 5e-4)',
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='epochs between checkpoints',
    )

    # KFAC Parameters
    parser.add_argument(
        '--kfac-inv-update-steps',
        type=int,
        default=10,
        help='iters between kfac inv ops (0 disables kfac) (default: 10)',
    )
    parser.add_argument(
        '--kfac-factor-update-steps',
        type=int,
        default=1,
        help='iters between kfac cov ops (default: 1)',
    )
    parser.add_argument(
        '--kfac-update-steps-alpha',
        type=float,
        default=10,
        help='KFAC update step multiplier (default: 10)',
    )
    parser.add_argument(
        '--kfac-update-steps-decay',
        nargs='+',
        type=int,
        default=None,
        help='KFAC update step decay schedule (default None)',
    )
    parser.add_argument(
        '--kfac-inv-method',
        action='store_true',
        default=False,
        help='Use inverse KFAC update instead of eigen (default False)',
    )
    parser.add_argument(
        '--kfac-factor-decay',
        type=float,
        default=0.95,
        help='Alpha value for covariance accumulation (default: 0.95)',
    )
    parser.add_argument(
        '--kfac-damping',
        type=float,
        default=0.003,
        help='KFAC damping factor (defaultL 0.003)',
    )
    parser.add_argument(
        '--kfac-damping-alpha',
        type=float,
        default=0.5,
        help='KFAC damping decay factor (default: 0.5)',
    )
    parser.add_argument(
        '--kfac-damping-decay',
        nargs='+',
        type=int,
        default=None,
        help='KFAC damping decay schedule (default None)',
    )
    parser.add_argument(
        '--kfac-kl-clip',
        type=float,
        default=0.001,
        help='KL clip (default: 0.001)',
    )
    parser.add_argument(
        '--kfac-skip-layers',
        nargs='+',
        type=str,
        default=[],
        help='Layer types to ignore registering with KFAC (default: [])',
    )
    parser.add_argument(
        '--kfac-colocate-factors',
        action='store_true',
        default=True,
        help='Compute A and G for a single layer on the same worker. ',
    )
    parser.add_argument(
        '--kfac-strategy',
        type=str,
        default='comm-opt',
        help='KFAC communication optimization strategy. One of comm-opt, '
        'mem-opt, or hybrid_opt. (default: comm-opt)',
    )
    parser.add_argument(
        '--kfac-grad-worker-fraction',
        type=float,
        default=0.25,
        help='Fraction of workers to compute the gradients '
        'when using HYBRID_OPT (default: 0.25)',
    )

    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )
    # Set automatically by torch distributed launch
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='local rank for distributed training',
    )
    
    parser.add_argument(
        '--timestamp',
        type=str,
        default=datetime.datetime.now().strftime('%Y%m%d_%H%M'),
        help='timestamp for the experiment',
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default='cifar10_resnet',
        help='name of the experiment',
    )

    parser.add_argument(
        '--recover',
        action='store_true',
        default=False,
        help='recover from checkpoint',
    )
    
    parser.add_argument(
        '--not-kfac',
        action='store_true',
        default=False,
        help='disable kfac',
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        default="CIFAR10",
        help='name of the dataset',
    )

    parser.add_argument(
        '--train-com-method',
        type=str,
        default="ddp",
        help='communication method for training',
    )

    parser.add_argument(
        '--lr-scheduler-type',
        type=str,
        default="multi_step",
        help='lr scheduler type',
    )

    parser.add_argument(
        '--optimizer-type',
        type=str,
        default="sgd",
        help='optimizer type , sgd or adamw',
    )

    parser.add_argument(
        '--degree-noniid',
        type=float,
        default=0,
        help='degree of non-iid data distribution',
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args