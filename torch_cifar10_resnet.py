"""Cifar10 and ResNet training script."""
from __future__ import annotations

import argparse
import ast
import datetime
import os
import time

import torch
import torch.distributed as dist
from torch.utils import collect_env
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import examples.vision.cifar_resnet as models
import examples.vision.datasets as datasets
import examples.vision.engine as engine
import examples.vision.optimizers as optimizers
from examples.utils import save_checkpoint

import kfac.mischief as mischief

try:
    from torch.cuda.amp import GradScaler

    TORCH_FP16 = True
except ImportError:
    TORCH_FP16 = False

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', 0))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', 0))

def string_to_list(input_str):
    try:
        # 使用ast.literal_eval安全地评估字符串
        result = ast.literal_eval(input_str)
        # 检查结果是否为列表
        if not isinstance(result, list):
            raise argparse.ArgumentTypeError(f"Input is not a list: {input_str}")
        # 检查列表元素是否全部为数字
        if not all(isinstance(item, int) for item in result):
            raise argparse.ArgumentTypeError(f"List items are not all integers: {input_str}")
        return result
    except (ValueError, SyntaxError):
        # 捕获评估过程中的错误，并抛出ArgumentTypeError
        raise argparse.ArgumentTypeError(f"Invalid input format: {input_str}")

def parse_args() -> argparse.Namespace:
    """Get cmd line args."""
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/work/NBB/yu_mingzhe/kfac-pytorch/data',
        metavar='D',
        help='directory to download cifar10 dataset to',
    )
    parser.add_argument(
        '--log-dir',
        default=f'/work/NBB/yu_mingzhe/kfac-pytorch/logs/torch_cifar10',
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
        default='resnet32',
        help='ResNet model to use [20, 32, 56]',
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
        default=20,
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
        '--disconnect-node-number',
        type=int,
        default=1,
        help='disconnect-node',
    )
    parser.add_argument(
        '--disconnect-terms',
        type=int,
        default=1,
        help='disconnect-terms',
    )
    parser.add_argument(
        '--connect-terms',
        type=int,
        default=9,
        help='connect-terms',
    )
    parser.add_argument(
        '--times',
        type=int,
        default=0,
        help='connect-terms',
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    mischief.DISCONNECTED_NODE_NUM = args.disconnect_node_number
    mischief.DISCONNECT_TERM = args.disconnect_terms
    mischief.CONNECT_TERM = args.connect_terms
    mischief.TIMES = args.times

    return args


def main() -> None:
    """Main train and eval function."""
    args = parse_args()
    timeout = datetime.timedelta(seconds=20)
    #dist.init_process_group("nccl",world_size=ompi_world_size,rank=ompi_world_rank,init_method='file:///work/NBB/yu_mingzhe/share_files/dist_file')
    dist.init_process_group( "nccl",timeout=timeout)

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    args.base_lr = (
        args.base_lr * dist.get_world_size() * args.batches_per_allreduce
    )
    args.verbose = dist.get_rank() == 0

    if args.verbose:
        print('Collecting env info...')
        print(collect_env.get_pretty_env_info())
        print(f"disconnect node list {mischief.DISCONNECTED_NODE}")
        print()

    for r in range(torch.distributed.get_world_size()):
        if r == torch.distributed.get_rank():
            print(
                f'Global rank {torch.distributed.get_rank()} initialized: '
                f'local_rank = {args.local_rank}, '
                f'world_size = {torch.distributed.get_world_size()}',
            )
        torch.distributed.barrier()

    train_sampler, train_loader, _, val_loader = datasets.get_cifar(args)
    model = models.get_model(args.model)

    device = 'cpu' if not args.cuda else 'cuda'
    model.to(device)

    if args.verbose:
        summary(model, (args.batch_size, 3, 32, 32), device=device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
    )
    mischief.add_hook_to_model(model)

    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)

    rank = dist.get_rank()
    board_dir = board_dir = (f"/work/NBB/yu_mingzhe/experiments/runs/"
                 f"cifar10_resnet32/p{dist.get_world_size()}n_{mischief.DISCONNECTED_NODE_NUM}e_"
                 f"{mischief.CONNECT_TERM}ct_{mischief.DISCONNECT_TERM}dct_{args.times}")
    if rank ==0:
        args.log_writer = SummaryWriter(board_dir)
    else:
        args.log_writer = None 

    args.resume_from_epoch = 0
    """
    # no check-point
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break
    """

    scaler = None
    if args.fp16:
        if not TORCH_FP16:
            raise ValueError(
                'The installed version of torch does not '
                'support torch.cuda.amp fp16 training. This '
                'requires torch version >= 1.16',
            )
        scaler = GradScaler()
    args.grad_scaler = scaler

    (
        optimizer,
        preconditioner,
        (lr_scheduler, kfac_scheduler),
    ) = optimizers.get_optimizer(
        model,
        args,
    )
    if args.verbose:
        print(preconditioner)
    loss_func = torch.nn.CrossEntropyLoss()

    if args.resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        map_location = {'cuda:0': f'cuda:{args.local_rank}'}
        checkpoint = torch.load(filepath, map_location=map_location)
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['lr_scheduler'] is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if (
            checkpoint['preconditioner'] is not None
            and preconditioner is not None
        ):
            preconditioner.load_state_dict(checkpoint['preconditioner'])

    start = time.time()

    for epoch in range(args.resume_from_epoch + 1, args.epochs + 1):
        mischief.term = epoch-1
        mischief.DISCONNECTED_NODE = mischief.shuffle_disconneted_node_list(dist.get_world_size(),args.epochs)
        if args.verbose and not mischief.is_connected_in_this_term():
             print(f"{mischief.DISCONNECTED_NODE} in term {mischief.term}")
        engine.train(
            epoch,
            model,
            optimizer,
            preconditioner,
            loss_func,
            train_sampler,
            train_loader,
            args,
        )
        engine.test(epoch, model, loss_func, val_loader, args)
        lr_scheduler.step()
        if kfac_scheduler is not None:
            kfac_scheduler.step(step=epoch)
        '''
        if (
            epoch > 0
            and epoch % args.checkpoint_freq == 0
            and dist.get_rank() == 0
        ):
            # Note: save model.module b/c model may be Distributed wrapper
            # so saving the underlying model is more generic
            save_checkpoint(
                model.module,
                optimizer,
                preconditioner,
                lr_scheduler,
                args.checkpoint_format.format(epoch=epoch),
            )
        '''

    dist.destroy_process_group()
    if args.verbose:
        print(
            '\nTraining time: {}'.format(
                datetime.timedelta(seconds=time.time() - start),
            ),
        )
        with open("/work/NBB/yu_mingzhe/kfac-pytorch/log",mode='a+') as f:
            f.write('\nTraining time: {}'.format(datetime.timedelta(seconds=time.time() - start)))

if __name__ == '__main__':
    main()
