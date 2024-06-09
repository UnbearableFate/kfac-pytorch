
import datetime
import time

import torch.distributed as dist
import kfac.rpc_distributed as rpc_distributed
from kfac.preconditioner import KFACPreconditioner
from my_module.custom_resnet import MLP
import logging
logging.basicConfig(level=logging.NOTSET)

timeout = datetime.timedelta(seconds=30)
dist.init_process_group("gloo", timeout=timeout)
model = MLP(num_hidden_layers=1)
preconditioner = KFACPreconditioner(model,update_factors_in_hook = False)
rpc_communicator = rpc_distributed.KFacRPCCommunicator(world_size=dist.get_world_size(), rank=dist.get_rank(), preconditioner=preconditioner)
rpc_distributed.global_communicator = rpc_communicator
rank = dist.get_rank()
world_size = dist.get_world_size()
for t in range(2):
    rpc_communicator.update_self_t()
    rpc_communicator.send_kfac_factor((rank +t)%world_size,"layers.1","A")
    rpc_communicator.send_kfac_eigen_tensor("layers.1","G")
    time.sleep(1)
    if rank == 0:
        print(repr(rpc_communicator))
time.sleep(3)
if rank == 0:
    print(repr(rpc_communicator))
rpc_communicator.shutdown()
dist.destroy_process_group()