import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING
from mpi4py import MPI
import kfac.rpc_util.GraphConstruct as GraphConstruct
from kfac.rpc_util.common_util import flatten_tensor2model, compute_l2_norm
import random
from kfac.adsgds.common import ModelStore, rpc_work_name,RootModelAvgRPCCommunicator , compute_recv_weight_by_loss
import torch.distributed as dist
import time
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class SwiftManager(RootModelAvgRPCCommunicator):
    def __init__(self, rank: int, model: torch.nn.Module, rpc_communicator: 'KFacRPCCommunicator'):
        super().__init__(rank, model, rpc_communicator)
        self.graph = GraphConstruct.GraphConstruct(rank,self.origin_world_size, MPI.COMM_WORLD, 'clique-ring', 'swift', p = 0.15, num_c=3)
        self.sw = 1 - sum(self.graph.neighbor_weights)
        print(f"Rank {rank} has neighbors {self.graph.neighbor_list}")
        print(f"Rank {rank} has weights {self.graph.neighbor_weights}")
        print(f"Rank {rank} has sw {self.sw}")
        self.neighbor_model_buffers = {}
        self.neighbor_model_buffers[(self.rank+1)%self.origin_world_size] = ModelStore(self.local_model_store.flatten_tensor)
        self.neighbor_model_buffers[(self.rank-1)%self.origin_world_size] = ModelStore(self.local_model_store.flatten_tensor)

        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    """
    call this func after backward propagation
    """
    def process(self):
        self.update_local_flat_model()
        send_target = (self.rank + 1) % self.origin_world_size
        recv_from = (self.rank - 1) % self.origin_world_size
        send_op = dist.P2POp(dist.isend, self.local_model_store.flatten_tensor, send_target)
        recv_op = dist.P2POp(dist.irecv, self.neighbor_model_buffers[recv_from].flatten_tensor, recv_from)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()

model_avg_rpc_communicator: SwiftManager
    