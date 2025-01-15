import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING
from mpi4py import MPI
import kfac.rpc_util.GraphConstruct as GraphConstruct
from kfac.rpc_util.common_util import flatten_tensor2model, model2flatten_tensor
import random
from kfac.adsgds.layer_type_common import LayerwiseModelStore, RootModelAvgRPCCommunicator, rpc_work_name
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class AdpsgdManager(RootModelAvgRPCCommunicator):
    def __init__(self, rank, model: torch.nn.Module, rpc_communicator: 'KFacRPCCommunicator'):
        super().__init__(rank, model, rpc_communicator)
        self.graph = GraphConstruct.GraphConstruct(rank,self.origin_world_size, MPI.COMM_WORLD, 'clique-ring', 'swift', p = 0.15, num_c=3)
        self.neighbor_model_buffer = LayerwiseModelStore()
        self.neighbor_model_buffer.clone_model_store(self.local_model_store)
        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    """
    call this func after backward propagation
    """
    def adsgd_exchange_model(self):
        target_neighbor = random.choice(self.graph.neighbor_list)
        res = rpc.rpc_sync(
            to=rpc_work_name(target_neighbor),
            func= exchange_model_param,
            args=(*self.local_model_store.getData(), self.rank)
        )
        if res is not None and len(res) == 3:
            self.local_model_store.aggrate_with_another_model(res[0], 0.5)
        else:
            print(f"Rank {self.rank} get None from {target_neighbor}")

model_avg_rpc_communicator: AdpsgdManager

def exchange_model_param(data, from_term, from_loss, from_rank):
    global model_avg_rpc_communicator
    return model_avg_rpc_communicator.local_model_store.getData()