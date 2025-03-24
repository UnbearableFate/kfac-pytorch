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
    parser = argparse.ArgumentParser(description='Unified PyTorch Training')

    # ==== General Settings ====
    parser.add_argument('--data-dir', default='/tmp/cifar10', help='Directory for dataset')
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', help='Alternative dataset path')
    parser.add_argument('--dataset-name', default='CIFAR10', help='Dataset name (e.g., CIFAR10, ImageNet)')
    parser.add_argument('--log-dir', default='./logs', help='Log directory')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument('--amp', action='store_true', help='Use AMP mixed precision')
    parser.add_argument('--use-deterministic-algorithms', action='store_true', help='Use deterministic algorithms')

    # ==== Model Settings ====
    parser.add_argument('--model', default='resnet18', help='Model name')
    parser.add_argument('--layers', type=int, default=34, help='Number of layers in ResNet')
    parser.add_argument('--weights', default=None, help='Pretrained weights to load')

    # ==== Training Settings ====
    parser.add_argument('--batch-size', default=128, type=int, help='Training batch size')
    parser.add_argument('--val-batch-size', default=128, type=int, help='Validation batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--base-lr', default=0.1, type=float, help='Base learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Optimizer momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--optimizer-type', default='sgd', choices=['sgd', 'adamw'], help='Optimizer type')
    parser.add_argument('--clip-grad-norm', default=None, type=float, help='Gradient clipping')

    # ==== LR Scheduler Settings ====
    parser.add_argument('--lr-scheduler-type', default='steplr', help='LR scheduler type')
    parser.add_argument('--lr-step-size', default=30, type=int, help='Step size for LR scheduler')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='LR decay factor')
    parser.add_argument('--lr-min', default=0.0, type=float, help='Minimum LR')
    parser.add_argument('--warmup-epochs', default=0, type=int, help='Number of warmup epochs')
    parser.add_argument('--warmup-method', default='constant', help='LR warmup method')
    parser.add_argument('--warmup-decay', default=0.01, type=float, help='LR warmup decay factor')

    # ==== Data Augmentation ====
    parser.add_argument('--label-smoothing', default=0.0, type=float, help='Label smoothing factor')
    parser.add_argument('--mixup-alpha', default=0.0, type=float, help='Mixup alpha')
    parser.add_argument('--cutmix-alpha', default=0.0, type=float, help='Cutmix alpha')
    parser.add_argument('--auto-augment', default=None, help='Auto augment policy')
    parser.add_argument('--ra-magnitude', default=9, type=int, help='RandAugment magnitude')
    parser.add_argument('--augmix-severity', default=3, type=int, help='AugMix severity level')
    parser.add_argument('--random-erase', default=0.0, type=float, help='Random erase probability')

    # ==== Distributed Training ====
    parser.add_argument('--backend', default='nccl', help='Distributed backend')
    parser.add_argument('--world-size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training')
    parser.add_argument('--dist-url', default='env://', help='Distributed URL')
    parser.add_argument('--sync-bn', action='store_true', help='Use SyncBatchNorm')
    parser.add_argument('--train-com-method', default='ddp', help='Distributed training communication method')

    # ==== EMA Settings ====
    parser.add_argument('--model-ema', action='store_true', help='Enable model EMA')
    parser.add_argument('--model-ema-steps', default=32, type=int, help='EMA update steps')
    parser.add_argument('--model-ema-decay', default=0.99998, type=float, help='EMA decay rate')

    # ==== KFAC Specific Parameters ====
    parser.add_argument('--kfac-inv-update-steps', type=int, default=10, help='Steps between KFAC inverse updates')
    parser.add_argument('--kfac-factor-update-steps', type=int, default=1, help='Steps between KFAC factor updates')
    parser.add_argument('--kfac-update-steps-alpha', type=float, default=10, help='Multiplier for KFAC update steps')
    parser.add_argument('--kfac-update-steps-decay', nargs='+', type=int, default=None, help='Decay schedule for KFAC update steps')
    parser.add_argument('--kfac-inv-method', action='store_true', help='Use inverse KFAC method')
    parser.add_argument('--kfac-factor-decay', type=float, default=0.95, help='Factor decay for KFAC covariance accumulation')
    parser.add_argument('--kfac-damping', type=float, default=0.003, help='KFAC damping factor')
    parser.add_argument('--kfac-damping-alpha', type=float, default=0.5, help='Decay factor for KFAC damping')
    parser.add_argument('--kfac-damping-decay', nargs='+', type=int, default=None, help='Damping decay schedule for KFAC')
    parser.add_argument('--kfac-kl-clip', type=float, default=0.001, help='KL clip value for KFAC')
    parser.add_argument('--kfac-skip-layers', nargs='+', type=str, default=[], help='Layers to skip for KFAC')
    parser.add_argument('--kfac-colocate-factors', action='store_true', default=True, help='Colocate KFAC factors')
    parser.add_argument('--kfac-strategy', type=str, default='comm-opt', help='KFAC communication optimization strategy')
    parser.add_argument('--kfac-grad-worker-fraction', type=float, default=0.25, help='Gradient worker fraction for KFAC hybrid optimization')
    parser.add_argument('--not-kfac', action='store_true', default=False, help='Disable KFAC')

    # ==== Miscellaneous ====
    parser.add_argument('--print-freq', default=10, type=int, help='Print frequency')
    parser.add_argument('--checkpoint-format', default='checkpoint_{epoch}.pth.tar', help='Checkpoint format')
    parser.add_argument('--checkpoint-freq', default=10, type=int, help='Checkpoint frequency')
    parser.add_argument('--recover', action='store_true', help='Recover from checkpoint')
    parser.add_argument('--resume', default='', help='Checkpoint resume path')
    parser.add_argument('--timestamp', default=datetime.datetime.now().strftime('%Y%m%d_%H%M'), help='Timestamp for the experiment')
    parser.add_argument('--experiment-name', default='experiment', help='Experiment name')
    parser.add_argument('--degree-noniid', type=float, default=0, help='Degree of non-IID distribution')

    args = parser.parse_args()

    # Automatically set local_rank if available
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args