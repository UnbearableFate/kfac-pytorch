import kfac
import torch.distributed as dist
from my_module.custom_resnet import MLP
import kfac.rpc_distributed as rpc_distributed

import logging

dist.init_process_group("gloo")
if dist.get_rank() == 0:
    logging.basicConfig(level=logging.NOTSET)
mlp = MLP(num_hidden_layers=7,hidden_size=128)
preconditioner = kfac.preconditioner.KFACPreconditioner(model=mlp)
rpc_distributed.KFacRPCCommunicator(dist.get_world_size(),dist.get_rank(),preconditioner,mlp)
dist.barrier()
dist.destroy_process_group()