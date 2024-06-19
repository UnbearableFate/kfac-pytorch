
import datetime
import time

import torch.distributed as dist
import kfac.rpc_distributed as rpc_distributed
from kfac.preconditioner import KFACPreconditioner
from my_module.custom_resnet import MLP ,ResNetForCIFAR10
import logging
logging.basicConfig(level=logging.NOTSET)

timeout = datetime.timedelta(seconds=30)
dist.init_process_group("gloo", timeout=timeout)
model = ResNetForCIFAR10(18)
preconditioner = KFACPreconditioner(model)
rpc_communicator = rpc_distributed.KFacRPCCommunicator(world_size=dist.get_world_size(), rank=dist.get_rank(), preconditioner=preconditioner ,model=model)
rpc_distributed.global_communicator = rpc_communicator
#rpc_communicator.send_model_param(model)
rank = dist.get_rank()
world_size = dist.get_world_size()

for t in range(4):
    rpc_communicator.update_self_t()
    rpc_communicator.send_model_param()

rpc_communicator.shutdown()
dist.destroy_process_group()